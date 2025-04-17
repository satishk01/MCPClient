"# MCPClient" 


###### to connect and run
To connect to the remote playwright-mcp Docs:
npx ts-node index.ts --remote "playwright-mcp Docs"


###### second sample

export GITHUB_USERNAME=your_github_username
export GITHUB_TOKEN=your_github_token

npx ts-node index_auth.ts --remote "LLMsample Docs" --username ${GITHUB_USERNAME} --password ${GITHUB_TOKEN}
