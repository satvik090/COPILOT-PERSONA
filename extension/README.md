# CopilotPersona VS Code Extension

## Install

1. Open the `extension` folder in VS Code.
2. Run `npm install -g vsce` if you do not already have the VS Code packaging tool.
3. Package the extension with `vsce package`.
4. In VS Code, open the Extensions view, choose `Install from VSIX...`, and select the generated `.vsix` file.
5. Start the backend with Docker Compose so the extension can reach `http://localhost:8000`.

## Command

- Run `CopilotPersona: Ping API` from the Command Palette to verify the extension can reach the backend.
