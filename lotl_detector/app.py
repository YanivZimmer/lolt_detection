"""
Chainlit app for interactive LOTL detection.
"""
import chainlit as cl
import json
from pathlib import Path
from typing import Dict, Any
import traceback

from ensemble import LOTLEnsemble
from data_loader import sanitize_event_for_inference


# Global model instance
ensemble_model = None


@cl.on_chat_start
async def start():
    """Initialize the app and load the model."""
    await cl.Message(
        content="Loading LOTL detection model...",
    ).send()
    
    global ensemble_model
    
    # Try to load the model
    model_dir = Path("models")
    if model_dir.exists() and (model_dir / "random_forest.pkl").exists():
        try:
            ensemble_model = LOTLEnsemble()
            ensemble_model.load(str(model_dir))
            await cl.Message(
                content="✅ Model loaded successfully! You can now analyze Sysmon events.",
            ).send()
        except Exception as e:
            await cl.Message(
                content=f"❌ Error loading model: {str(e)}\n\nPlease train the model first using `make train`",
            ).send()
            ensemble_model = None
    else:
        await cl.Message(
            content="⚠️ Model not found. Please train the model first using `make train`",
        ).send()
        ensemble_model = None
    
    # Show example
    await cl.Message(
        content="""
## How to use:

1. **Paste a JSON event** - Copy a Sysmon event from your dataset or create one
2. **Or use the example below** - Click the example button

Example event format:
```json
{
  "EventID": 1,
  "EventTime": "2020-05-01 22:56:35",
  "UtcTime": "2020-05-01T22:56:35.000Z",
  "Hostname": "UTICA.dmevals.local",
  "Channel": "Microsoft-Windows-Sysmon/Operational",
  "SourceImage": "C:WindowsSystem32cmd.exe",
  "CommandLine": "cmd.exe /c wevtutil cl Application",
  "ProcessId": "1150",
  "ProcessGUID": "{6bbf237a-047e-5eac-047e-00000000047e}",
  "ParentImage": "C:Windowsexplorer.exe",
  "ParentProcessId": "420",
  "ParentProcessGUID": "{6bbf237a-01a4-5eac-01a4-0000000001a4}",
  "ParentCommandLine": "C:Windowsexplorer.exe",
  "User": "NT AUTHORITYNETWORK SERVICE",
  "Domain": "NT AUTHORITY",
  "AccountName": "NETWORK SERVICE",
  "IntegrityLevel": "Medium",
  "CurrentDirectory": "C:WindowsSystem32\"",
  "LogonId": "0x3e7"
}

```
        """,
    ).send()


@cl.on_message
async def main(message: cl.Message):
    """Process incoming messages."""
    global ensemble_model
    
    if ensemble_model is None:
        await cl.Message(
            content="❌ Model not loaded. Please train the model first using `make train`",
        ).send()
        return
    
    try:
        # Parse JSON event
        event_text = message.content.strip()
        
        # Try to extract JSON from markdown code blocks first
        if '```json' in event_text:
            json_start = event_text.find('```json') + 7
            json_end = event_text.find('```', json_start)
            event_text = event_text[json_start:json_end].strip()
        elif '```' in event_text:
            json_start = event_text.find('```') + 3
            json_end = event_text.find('```', json_start)
            if json_end > json_start:
                event_text = event_text[json_start:json_end].strip()
        
        # Parse JSON (handle escaped backslashes properly)
        event = json.loads(event_text)
        
        # Sanitize event (remove metadata fields)
        event = sanitize_event_for_inference(event)
        
        # Get prediction with explanations (RF + optionally LLM)
        use_llm = False  # Can be made configurable
        results = ensemble_model.predict_with_explanation([event], use_llm_explanation=use_llm)
        result = results[0]
        
        # Format response
        prediction = result['prediction']
        confidence = result['confidence']
        rf_explanation = result.get('rf_explanation', result.get('explanation', 'No explanation available'))
        llm_explanation = result.get('llm_explanation')
        
        # Determine color/icon
        if prediction == 'malicious':
            icon = "🔴"
            color = "red"
        else:
            icon = "🟢"
            color = "green"
        
        # Create response message
        response = f"""
{icon} **Prediction: {prediction.upper()}**

**Confidence:** {confidence:.1%}

---

## Features Explanation

{rf_explanation}

"""
        
        # Add LLM explanation if available
        if llm_explanation:
            response += f"""
---

## LLM Reasoning Explanation

{llm_explanation}

"""
        
        response += f"""
---

**Event Details:**
- Command: `{event.get('CommandLine', 'N/A')[:100]}`
- Image: `{event.get('Image', event.get('SourceImage', 'N/A'))}`
- User: `{event.get('User', 'N/A')}`
- Integrity Level: `{event.get('IntegrityLevel', 'N/A')}`
"""
        
        await cl.Message(
            content=response,
        ).send()
        
    except json.JSONDecodeError as e:
        await cl.Message(
            content=f"❌ Invalid JSON format. Please provide a valid JSON event.\n\nError: {str(e)}",
        ).send()
    except Exception as e:
        await cl.Message(
            content=f"❌ Error processing event: {str(e)} in {traceback.format_exc()}",
        ).send()


@cl.action_callback("example_event")
async def on_action(action):
    """Handle example event action."""
    example_event = {
        "EventID": 1,
        "EventTime": "2025-10-10 08:00:30",
        "CommandLine": "powershell.exe -enc SQBuAHYAbwBrAGUALQBXAGUAYgBSAGUAcQB1AGUAcwB0AA==",
        "Image": "C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe",
        "User": "CORP\\jsmith",
        "IntegrityLevel": "Medium"
    }
    
    await cl.Message(
        content=f"Example event:\n```json\n{json.dumps(example_event, indent=2)}\n```",
    ).send()

