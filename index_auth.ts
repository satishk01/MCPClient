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
import { spawn } from "child_process";

dotenv.config();

// Optional: AWS region, with default
const AWS_REGION = process.env.AWS_REGION || "us-east-1";

interface MCPServerConfig {
  command?: string;
  args?: string[];
  url?: string;
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
  async connectToRemoteServer(serverName: string, configPath?: string, username?: string, password?: string) {
    // Load the configuration
    const config = await this.loadConfig(configPath);
    
    if (!config.mcpServers || !config.mcpServers[serverName]) {
      throw new Error(`Server "${serverName}" not found in configuration`);
    }
    
    const serverConfig = config.mcpServers[serverName];
    console.log(`Connecting to remote server: ${serverName}`);
    
    // Handle the URL format (new configuration style)
    if (serverConfig.url) {
      // If no username/password provided, prompt for them
      const githubUsername = username || await this.promptForInput("GitHub Username: ");
      const githubPassword = password || await this.promptForInput("GitHub Password/Token: ", true);
      
      console.log(`Using mcp-remote to connect to: ${serverConfig.url}`);
      
      // Use mcp-remote with username and password
      this.transport = new StdioClientTransport({
        command: "npx",
        args: ["mcp-remote", serverConfig.url],
        env: {
          ...process.env,
          GITHUB_USERNAME: githubUsername,
          GITHUB_TOKEN: githubPassword
        }
      });
    } 
    // Handle the command/args format (old configuration style)
    else if (serverConfig.command && serverConfig.args) {
      console.log(`Command: ${serverConfig.command} ${serverConfig.args.join(' ')}`);
      
      this.transport = new StdioClientTransport({
        command: serverConfig.command,
        args: serverConfig.args,
      });
    } else {
      throw new Error("Invalid server configuration: missing url or command/args");
    }
    
    await this.connectWithTransport();
  }

  // Helper method to prompt for user input (for credentials)
  private async promptForInput(prompt: string, isPassword = false): Promise<string> {
    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
    });
    
    try {
      // If it's a password and we're in a TTY environment, we can try to hide the input
      if (isPassword && process.stdin.isTTY) {
        // This is a simple implementation - in a real app you might want to use a package like 'prompt'
        process.stdout.write(prompt);
        return new Promise((resolve) => {
          const stdin = process.stdin;
          stdin.setRawMode(true);
          stdin.resume();
          let password = '';
          
          stdin.on('data', (data) => {
            const char = data.toString();
            
            // Handle backspace
            if (char === '\b' || char === '\x7f') {
              if (password.length > 0) {
                password = password.substring(0, password.length - 1);
                process.stdout.write('\b \b');
              }
              return;
            }
            
            // Handle enter (carriage return)
            if (char === '\r' || char === '\n') {
              stdin.setRawMode(false);
              stdin.pause();
              console.log(''); // New line after password
              resolve(password);
              return;
            }
            
            // Add character to password
            password += char;
            process.stdout.write('*');
          });
        });
      } else {
        // Regular prompt for non-password or non-TTY
        return await rl.question(prompt);
      }
    } finally {
      rl.close();
    }
  }

  // Helper method to establish connection with the transport
  private async connectWithTransport() {
    if (!this.transport) {
      throw new Error("Transport not initialized");
    }
    
    try {
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
    } catch (error) {
      console.error("Failed to connect to MCP server:", error);
      if (error instanceof Error) {
        throw new Error(`Connection failed: ${error.message}`);
      } else {
        throw new Error(`Connection failed: ${String(error)}`);
      }
    }
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
  For remote server:   node index.ts --remote <server_name> [--config <config_path>] [--username <github_username>] [--password <github_password>]
  
Examples:
  node index.ts --local ./server.py
  node index.ts --remote "LLMsample Docs"
  node index.ts --remote "LLMsample Docs" --username myusername --password mytoken
      `);
      return;
    }
    
    if (args[0] === "--local" && args.length > 1) {
      await mcpClient.connectToLocalServer(args[1]);
    } else if (args[0] === "--remote" && args.length > 1) {
      const serverName = args[1];
      const configPath = args.indexOf("--config") > -1 ? args[args.indexOf("--config") + 1] : undefined;
      const username = args.indexOf("--username") > -1 ? args[args.indexOf("--username") + 1] : undefined;
      const password = args.indexOf("--password") > -1 ? args[args.indexOf("--password") + 1] : undefined;
      
      await mcpClient.connectToRemoteServer(serverName, configPath, username, password);
    } else {
      console.error("Invalid arguments. Use --help for usage information.");
      return;
    }
    
    await mcpClient.chatLoop();
  } catch (error) {
    console.error("Error:", error);
  } finally {
    await mcpClient.cleanup();
    process.exit(0);
  }
}

main();