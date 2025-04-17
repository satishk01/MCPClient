// aws bedrock & anthropic sdk
import { BedrockRuntimeClient, InvokeModelCommand } from "@aws-sdk/client-bedrock-runtime";
import {
  MessageParam,
  Tool,
} from "@anthropic-ai/sdk/resources/messages/messages.mjs";

// mcp sdk
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";

import dotenv from "dotenv";
import readline from "readline/promises";
import fs from "fs";
import path from "path";

dotenv.config();

// Optional: AWS region, with default
const AWS_REGION = process.env.AWS_REGION || "us-east-1";

interface MCPServerConfig {
  command: string;
  args: string[];
}

interface MCPConfig {
  mcpServers: {
    [key: string]: MCPServerConfig;
  };
}

class MCPClient {
  private mcp: Client;
  private bedrock: BedrockRuntimeClient;
  private transport: StdioClientTransport | null = null;
  private tools: Tool[] = [];

  constructor() {
    // Initialize AWS Bedrock client using IAM role
    this.bedrock = new BedrockRuntimeClient({ 
      region: AWS_REGION
      // No explicit credentials - will use IAM role
    });
    this.mcp = new Client({ name: "mcp-client-cli", version: "1.0.0" });
  }

  // Connect to a local MCP server script
  async connectToLocalServer(serverScriptPath: string) {
    const isJs = serverScriptPath.endsWith(".js");
    const isPy = serverScriptPath.endsWith(".py");
    if (!isJs && !isPy) {
      throw new Error("Server script must be a .js or .py file");
    }
    const command = isPy
      ? process.platform === "win32"
        ? "python"
        : "python3"
      : process.execPath;

    this.transport = new StdioClientTransport({
      command,
      args: [serverScriptPath],
    });
    
    await this.connectWithTransport();
  }

  // Connect to a remote MCP server using configuration
  async connectToRemoteServer(serverName: string, configPath?: string) {
    // Load the configuration
    const config = await this.loadConfig(configPath);
    
    if (!config.mcpServers || !config.mcpServers[serverName]) {
      throw new Error(`Server "${serverName}" not found in configuration`);
    }
    
    const serverConfig = config.mcpServers[serverName];
    console.log(`Connecting to remote server: ${serverName}`);
    console.log(`Command: ${serverConfig.command} ${serverConfig.args.join(' ')}`);
    
    this.transport = new StdioClientTransport({
      command: serverConfig.command,
      args: serverConfig.args,
    });
    
    await this.connectWithTransport();
  }

  // Helper method to establish connection with the transport
  private async connectWithTransport() {
    if (!this.transport) {
      throw new Error("Transport not initialized");
    }
    
    await this.mcp.connect(this.transport);

    // Register tools
    const toolsResult = await this.mcp.listTools();
    this.tools = toolsResult.tools.map((tool) => {
      return {
        name: tool.name,
        description: tool.description,
        input_schema: tool.inputSchema,
      };
    });

    console.log(
      "Connected to server with tools:",
      this.tools.map(({ name }) => name)
    );
  }

  // Load configuration from file
  private async loadConfig(configPath?: string): Promise<MCPConfig> {
    const defaultPaths = [
      path.join(process.cwd(), '.mcp.json'),
      path.join(process.cwd(), 'mcp-config.json'),
      path.join(process.env.HOME || process.env.USERPROFILE || '', '.mcp', 'config.json')
    ];
    
    const paths = configPath ? [configPath] : defaultPaths;
    
    for (const p of paths) {
      try {
        if (fs.existsSync(p)) {
          const content = await fs.promises.readFile(p, 'utf-8');
          return JSON.parse(content);
        }
      } catch (error) {
        console.warn(`Failed to read config from ${p}:`, error);
      }
    }
    
    // If no configuration file found, return a default config
    return { mcpServers: {} };
  }

  // Helper function to call Bedrock Claude model
  async invokeClaudeModel(messages: MessageParam[], tools?: Tool[]) {
    const payload = {
      anthropic_version: "bedrock-2023-05-31",
      max_tokens: 1000,
      messages,
      tools: tools || [],
    };
    
    const command = new InvokeModelCommand({
      modelId: "anthropic.claude-3-sonnet-20240229-v1:0", // Bedrock model ID for Claude 3 Sonnet
      body: JSON.stringify(payload),
      contentType: "application/json",
    });

    try {
      const response = await this.bedrock.send(command);
      const responseBody = JSON.parse(Buffer.from(response.body).toString());
      return responseBody;
    } catch (error) {
      console.error("Error invoking Bedrock model:", error);
      if (error instanceof Error) {
        throw new Error(`Failed to invoke Bedrock model: ${error.message}`);
      } else {
        throw new Error(`Failed to invoke Bedrock model: ${String(error)}`);
      }
    }
  }

  // Process query
  async processQuery(query: string) {
    // create messages array
    const messages: MessageParam[] = [
      {
        role: "user",
        content: query,
      },
    ];

    // Call Bedrock with tools
    const response = await this.invokeClaudeModel(messages, this.tools);

    // Process the response
    const finalText = [];
    const toolResults = [];

    // Process content from response
    for (const content of response.content) {
      if (content.type === "text") {
        finalText.push(content.text);
      } else if (content.type === "tool_use") {
        // If tool -> call the tool on mcp server
        const toolName = content.name;
        const toolArgs = content.input as { [x: string]: unknown } | undefined;
        const toolId = content.id;

        const result = await this.mcp.callTool({
          name: toolName,
          arguments: toolArgs,
        });
        toolResults.push(result);
        finalText.push(
          `[Calling tool ${toolName} with args ${JSON.stringify(toolArgs)}]`
        );
        
        // Add tool result as a user message
        messages.push({
          role: "assistant",
          content: [{ 
            type: "tool_use", 
            name: toolName, 
            input: toolArgs,
            id: toolId
          }]
        });
        messages.push({
          role: "user",
          content: [{ 
            type: "tool_result", 
            tool_use_id: toolId, 
            content: result.content as string 
          }]
        });

        // Get the follow-up response from Claude
        const followUpResponse = await this.invokeClaudeModel(messages, this.tools);
        
        if (followUpResponse.content[0].type === "text") {
          finalText.push(followUpResponse.content[0].text);
        }
      }
    }

    return finalText.join("\n");
  }

  async chatLoop() {
    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
    });

    try {
      console.log("\nMCP Client Started with AWS Bedrock (using IAM role authentication)!");
      console.log("Type your queries or 'quit' to exit.");

      while (true) {
        const message = await rl.question("\nQuery: ");
        if (message.toLowerCase() === "quit") {
          break;
        }
        const response = await this.processQuery(message);
        console.log("\n" + response);
      }
    } finally {
      rl.close();
    }
  }

  async cleanup() {
    if (this.mcp) {
      await this.mcp.close();
    }
  }
}

async function main() {
  const mcpClient = new MCPClient();
  
  try {
    // Parse command line arguments
    const args = process.argv.slice(2);
    
    if (args.length === 0 || args[0] === "--help" || args[0] === "-h") {
      console.log(`
Usage:
  For local server:    node index.ts --local <path_to_server_script>
  For remote server:   node index.ts --remote <server_name> [--config <config_path>]
  
Examples:
  node index.ts --local ./server.py
  node index.ts --remote "playwright-mcp Docs"
      `);
      return;
    }
    
    if (args[0] === "--local" && args.length > 1) {
      await mcpClient.connectToLocalServer(args[1]);
    } else if (args[0] === "--remote" && args.length > 1) {
      const configPath = args.indexOf("--config") > -1 ? args[args.indexOf("--config") + 1] : undefined;
      await mcpClient.connectToRemoteServer(args[1], configPath);
    } else {
      console.error("Invalid arguments. Use --help for usage information.");
      return;
    }
    
    await mcpClient.chatLoop();
  } finally {
    await mcpClient.cleanup();
    process.exit(0);
  }
}

main();
