"""Product detection using Google Gemini Vision API."""

import os
import signal
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()


class TimeoutError(Exception):
    """Custom timeout exception."""


class GeminiVisionDetector:
    """Detects products and analyzes fill levels using Gemini Vision API."""
    
    TIMEOUT_SECONDS = 60
    DETECTION_PROMPT = """
You are an annotation engine. Look at the image and enumerate EVERY distinct, non-overlapping, real-world object fully or partially visible in the frame (e.g., bottles, tubes, jars, boxes). For each object, output EXACTLY one JSON object with this schema and key order:

[
  {
    "box_2d": [y0, x0, y1, x1],
    "label": "<concise object name + short attribute/description>",
    "percent_full": N,
    "is_low": true|false,
    "confidence": C
  },
  ...
]

STRICT RULES (follow exactly):
- OUTPUT FORMAT: Respond ONLY with a Markdown JSON code block. First line MUST be ```json and last line MUST be ```. No prose before/after. No explanations. No comments.
- JSON: Strict JSON; no trailing commas; double-quoted keys/strings; numbers only for percent_full/confidence; booleans are true/false (lowercase).
- SORTING: Sort objects by the top-left corner of the box (ascending x0, then ascending y0) for deterministic order.
- LABELING: Use generic names unless the brand text is clearly readable in the image; do NOT hallucinate brands or models.
- PERCENT FULL HEURISTICS (deterministic): 
  1) Transparent/Translucent containers: estimate fill line via visible liquid boundary within the mask's vertical extent. 
  2) Opaque containers with sight window: use the visible window only. 
  3) Fully opaque with no cues: use 100 if unopened-sealed cues; otherwise 50 by default. 
  4) Squeezables (tubes): infer from creases/flattening; if uncertain, default 40. 
  Clip the final value to [0,100] and round to nearest integer.
- CONFIDENCE: 0.9 when a clear fill boundary/window exists; 0.6 when inferred from shape/creases; 0.4 when no visual cues.
- BOX_2D: Use [y0, x0, y1, x1] normalized to 0–1000 (integers). Ensure y0<y1 and x0<x1. The box must tightly enclose the object's mask.

IF NOTHING IS DETECTED: Return an empty JSON array [] in the required code block.
"""
    
    def __init__(self, api_key: str):
        """Initialize the detector with API key."""
        self.client = genai.Client(api_key=api_key)
        self.start_time = None
    
    def timeout_handler(self, signum, frame):
        """Handle timeout signal."""
        raise TimeoutError(f"Request timed out after {self.TIMEOUT_SECONDS} seconds")
    
    def detect_products(self, image_path: str) -> str:
        """
        Detect products in an image and return structured JSON.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            JSON string with detected products and their properties
            
        Raises:
            TimeoutError: If the API request times out
            FileNotFoundError: If the image file doesn't exist
        """
        print("Reading image file...")
        try:
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        print(f"Image loaded: {len(image_bytes)} bytes")
        print("Sending request to Gemini API...")
        print("(This request asks for segmentation masks which can take a long time)")
        
        signal.signal(signal.SIGALRM, self.timeout_handler)
        signal.alarm(self.TIMEOUT_SECONDS)
        
        try:
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type='image/png',
                    ),
                    self.DETECTION_PROMPT
                ],
                config={
                    'temperature': 0.1,
                    'max_output_tokens': 4000,
                }
            )
            
            signal.alarm(0)
            elapsed = time.time() - self.start_time
            print(f"Response received in {elapsed:.2f} seconds")
            
            return response.text
            
        except Exception as e:
            signal.alarm(0)
            raise


def main():
    """Main execution function."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not found in environment variables")
        return
    
    detector = GeminiVisionDetector(api_key)
    
    try:
        detector.start_time = time.time()
        result = detector.detect_products('assets/multiple_products.png')
        
        print("\n" + "=" * 60)
        print(result)
        print("=" * 60)
        
    except TimeoutError as e:
        print(f"\n{str(e)}")
        print("The API request took too long. This is likely because:")
        print("- The prompt asked for complex binary segmentation masks")
        print("- The API may be rate-limited or slow")
        print("\nTry: Run the script again, or simplify the prompt further")
    
    except KeyboardInterrupt:
        print("\n\nScript interrupted by user")
    
    except Exception as e:
        print(f"\nError: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()

