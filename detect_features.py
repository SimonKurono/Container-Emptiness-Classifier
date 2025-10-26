from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
import time
import signal

# Add timeout handler
class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Request timed out after 60 seconds")

load_dotenv()

print("Initializing Gemini client...")
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

print("Reading image file...")
with open('multiple_products.png', 'rb') as f:
    image_bytes = f.read()
print(f"Image loaded: {len(image_bytes)} bytes")

print("Sending request to Gemini API...")
print("(This request asks for segmentation masks which can take a long time)")
start_time = time.time()

try:
    # Set timeout for 60 seconds
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(60)
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type='image/png',
            ),
            """
            You are an annotation engine. Look at the image and enumerate EVERY distinct, non-overlapping, real-world object fully or partially visible in the frame (e.g., bottles, tubes, jars, boxes). For each object, output EXACTLY one JSON object with this schema and key order:

[
  {
    "box_2d": [y0, x0, y1, x1],            // integers in [0,1000], normalized to image height/width; y before x; y0<x1 etc.
    "label": "<concise object name + short attribute/description>", // e.g., "shampoo bottle, white, pump"
    "percent_full": N,                      // integer 0–100 estimating how full the container is; 0 = empty, 100 = full; if not a container, use 100
    "is_low": true|false,                   // true iff percent_full < 25
    "confidence": C                         // float between 0.0 and 1.0 for the percent_full estimate
  },
  ...
]

STRICT RULES (follow exactly):
- OUTPUT FORMAT: Respond ONLY with a Markdown JSON code block. First line MUST be ```json and last line MUST be ```. No prose before/after. No explanations. No comments.
- JSON: Strict JSON; no trailing commas; double-quoted keys/strings; numbers only for percent_full/confidence; booleans are true/false (lowercase).
- SORTING: Sort objects by the top-left corner of the box (ascending x0, then ascending y0) for deterministic order.
- LABELING: Use generic names unless the brand text is clearly readable in the image; do NOT hallucinate brands or models.
- PERCENT FULL HEURISTICS (deterministic): 
  1) Transparent/Translucent containers: estimate fill line via visible liquid boundary within the mask’s vertical extent. 
  2) Opaque containers with sight window: use the visible window only. 
  3) Fully opaque with no cues: use 100 if unopened-sealed cues; otherwise 50 by default. 
  4) Squeezables (tubes): infer from creases/flattening; if uncertain, default 40. 
  Clip the final value to [0,100] and round to nearest integer.
- CONFIDENCE: 0.9 when a clear fill boundary/window exists; 0.6 when inferred from shape/creases; 0.4 when no visual cues.
- BOX_2D: Use [y0, x0, y1, x1] normalized to 0–1000 (integers). Ensure y0<y1 and x0<x1. The box must tightly enclose the object’s mask.

IF NOTHING IS DETECTED: Return an empty JSON array [] in the required code block.

            """
        ],
        config={
            'temperature': 0.1,
            'max_output_tokens': 4000,
        }
    )
    
    signal.alarm(0)  # Cancel timeout
    elapsed = time.time() - start_time
    print(f"Response received in {elapsed:.2f} seconds")
    print("\n" + "="*60)
    print(response.text)
    print("="*60)
    
except TimeoutError as e:
    elapsed = time.time() - start_time
    print(f"\n{str(e)}")
    print("The API request took too long. This is likely because:")
    print("- The prompt asked for complex binary segmentation masks")
    print("- The API may be rate-limited or slow")
    print("\nTry: Run the script again, or simplify the prompt further")
except KeyboardInterrupt:
    print("\n\nScript interrupted by user")
except Exception as e:
    elapsed = time.time() - start_time
    print(f"\nError after {elapsed:.2f} seconds: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

