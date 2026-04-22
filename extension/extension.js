"use strict";

const vscode = require("vscode");

let enabled = true;
let outputChannel;
let statusBarItem;
const patternContextByDocument = new Map();

function activate(context) {
  outputChannel = vscode.window.createOutputChannel("CopilotPersona");
  outputChannel.appendLine("Activating CopilotPersona extension.");

  enabled = getConfiguration().get("enabled", true);
  statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
  updateStatusBar();
  statusBarItem.show();

  const enableCommand = vscode.commands.registerCommand("copilotPersona.enable", async () => {
    await setEnabledState(true);
    vscode.window.showInformationMessage("CopilotPersona enabled.");
  });

  const disableCommand = vscode.commands.registerCommand("copilotPersona.disable", async () => {
    await setEnabledState(false);
    vscode.window.showInformationMessage("CopilotPersona disabled.");
  });

  const triggerIndexCommand = vscode.commands.registerCommand(
    "copilotPersona.triggerIndex",
    async () => {
      const serverUrl = getServerUrl();

      await vscode.window.withProgress(
        {
          location: vscode.ProgressLocation.Notification,
          title: "CopilotPersona: Re-indexing corpus",
          cancellable: false
        },
        async () => {
          const response = await fetch(`${serverUrl}/index/trigger`, {
            method: "POST",
            headers: {
              "Content-Type": "application/json"
            }
          });

          if (!response.ok) {
            throw new Error(`Index trigger failed with status ${response.status}`);
          }

          outputChannel.appendLine(`Triggered index job at ${serverUrl}/index/trigger.`);
          vscode.window.showInformationMessage("CopilotPersona indexing started.");
        }
      );
    }
  );

  const inlineProvider = vscode.languages.registerInlineCompletionItemProvider(
    { language: "python", scheme: "file" },
    {
      async provideInlineCompletionItems(document, position) {
        if (!enabled) {
          return undefined;
        }

        const fileText = document.getText();
        const cursorOffset = document.offsetAt(position);
        const contextText = fileText.slice(Math.max(0, cursorOffset - 500), cursorOffset);
        const currentPrompt = document.lineAt(position.line).text.slice(0, position.character);

        try {
          const result = await fetchPatternSummary(contextText, currentPrompt);
          if (!result || !result.pattern_summary) {
            patternContextByDocument.delete(document.uri.toString());
            await vscode.commands.executeCommand("setContext", "copilotPersona.patternSummary", "");
            return undefined;
          }

          const systemContext = `Personal coding patterns: ${result.pattern_summary}`;
          patternContextByDocument.set(document.uri.toString(), systemContext);
          await vscode.commands.executeCommand(
            "setContext",
            "copilotPersona.patternSummary",
            systemContext
          );
          outputChannel.appendLine(
            `[context] ${document.uri.fsPath}: ${systemContext}`
          );
        } catch (error) {
          outputChannel.appendLine(
            `[retrieve] ${document.uri.fsPath}: ${formatError(error)}`
          );
        }

        return undefined;
      }
    }
  );

  const configChange = vscode.workspace.onDidChangeConfiguration(async (event) => {
    if (event.affectsConfiguration("copilotPersona.enabled")) {
      enabled = getConfiguration().get("enabled", true);
      updateStatusBar();
      outputChannel.appendLine(`Configuration changed: enabled=${enabled}`);
    }

    if (event.affectsConfiguration("copilotPersona.serverUrl")) {
      outputChannel.appendLine(`Configuration changed: serverUrl=${getServerUrl()}`);
    }
  });

  context.subscriptions.push(
    outputChannel,
    statusBarItem,
    enableCommand,
    disableCommand,
    triggerIndexCommand,
    inlineProvider,
    configChange
  );
}

function deactivate() {
  if (outputChannel) {
    outputChannel.appendLine("Deactivating CopilotPersona extension.");
  }
}

function getConfiguration() {
  return vscode.workspace.getConfiguration("copilotPersona");
}

function getServerUrl() {
  const configured = getConfiguration().get("serverUrl", "http://localhost:8000");
  return String(configured).replace(/\/+$/, "");
}

async function setEnabledState(value) {
  enabled = value;
  await getConfiguration().update(
    "enabled",
    value,
    vscode.ConfigurationTarget.Global
  );
  updateStatusBar();
  outputChannel.appendLine(`Enabled state changed: ${enabled}`);
}

function updateStatusBar() {
  if (!statusBarItem) {
    return;
  }

  if (enabled) {
    statusBarItem.text = "$(person) CopilotPersona: ON";
    statusBarItem.tooltip = "Click to disable CopilotPersona.";
    statusBarItem.command = "copilotPersona.disable";
    return;
  }

  statusBarItem.text = "$(person) CopilotPersona: OFF";
  statusBarItem.tooltip = "Click to enable CopilotPersona.";
  statusBarItem.command = "copilotPersona.enable";
}

async function fetchPatternSummary(contextText, currentPrompt) {
  const serverUrl = getServerUrl();
  const response = await fetch(`${serverUrl}/retrieve`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      context: contextText,
      current_prompt: currentPrompt
    })
  });

  if (!response.ok) {
    throw new Error(`Retrieve request failed with status ${response.status}`);
  }

  const payload = await response.json();
  return {
    pattern_summary: payload.pattern_summary || "",
    chunks: Array.isArray(payload.chunks) ? payload.chunks : [],
    confidence_scores: payload.confidence_scores || {},
    retrieval_time_ms: typeof payload.retrieval_time_ms === "number"
      ? payload.retrieval_time_ms
      : 0,
    cache_hit: Boolean(payload.cache_hit)
  };
}

function formatError(error) {
  if (error instanceof Error) {
    return error.message;
  }

  return String(error);
}

module.exports = {
  activate,
  deactivate
};
