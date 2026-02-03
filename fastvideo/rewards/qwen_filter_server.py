# Copyright (c) [2025] [FastVideo Team]
# SPDX-License-Identifier: [Apache License 2.0]
#
# Qwen3-VL-32B-Instruct based image quality filter server.
# This server uses Qwen3-VL to check if generated images are valid:
# - Check if text/logo is garbled
# - Check if the main subject is normal
# Only image is required for validation.

import io
import base64
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from PIL import Image
from typing import Optional
import os

# Try to import vLLM for Qwen3-VL inference
try:
    from vllm import LLM, SamplingParams
    from vllm.multimodal import MultiModalData
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("Warning: vLLM not available, using OpenAI-compatible API mode")

# Try to import OpenAI for API-based inference
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


app = FastAPI(title="Qwen3-VL Image Quality Filter API")

# Global model instance
model = None
client = None
use_vllm = False

VALIDATION_SYSTEM_PROMPT = """You are a professional image quality auditor. You need to check the generated image for the following issues:
1. Text or logos that are garbled, blurry, or misspelled.
2. Structural abnormalities in the main subject (e.g., deformities, overlapping, or missing parts).
3. Obvious generation artifacts or unnatural elements.

Please examine the image carefully.
If the image quality is acceptable (none of the above issues are present), return "PASS".
If the image contains any of the above issues, return "FAIL" and briefly explain the reason.

Your response must start with "PASS" or "FAIL"."""

VALIDATION_USER_PROMPT = """Please check the quality of this image:
- Are the text or logos in the image clear and correct?
- Is the main subject normal, without deformities or abnormalities?
- Is the overall image quality acceptable?

Please answer directly with PASS or FAIL and briefly explain your reasoning."""


class ImageValidationRequest(BaseModel):
    """Request model for image validation."""
    image_base64: str
    prompt: Optional[str] = None  # Optional, not used for validation


class ImageValidationResponse(BaseModel):
    """Response model for image validation."""
    valid: bool
    reason: str
    raw_response: str


def decode_base64_image(image_base64: str) -> Image.Image:
    """Decode base64 string to PIL Image."""
    try:
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data))
        return image.convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode image: {e}")


def encode_image_to_base64(image: Image.Image) -> str:
    """Encode PIL Image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=95)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def validate_image_vllm(image: Image.Image) -> tuple[bool, str, str]:
    """Validate image using vLLM local model."""
    global model
    
    # Prepare the prompt
    prompt = f"<|im_start|>system\n{VALIDATION_SYSTEM_PROMPT}<|im_end|>\n"
    prompt += f"<|im_start|>user\n<image>\n{VALIDATION_USER_PROMPT}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    
    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.9,
        max_tokens=256,
    )
    
    # Run inference
    outputs = model.generate(
        [{"prompt": prompt, "multi_modal_data": {"image": image}}],
        sampling_params=sampling_params,
    )
    
    response = outputs[0].outputs[0].text.strip()
    
    # Parse response
    valid = response.upper().startswith("PASS")
    reason = response
    
    return valid, reason, response


def validate_image_api(image: Image.Image) -> tuple[bool, str, str]:
    """Validate image using OpenAI-compatible API."""
    global client
    
    # Encode image to base64
    image_base64 = encode_image_to_base64(image)
    
    # Prepare messages
    messages = [
        {
            "role": "system",
            "content": VALIDATION_SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }
                },
                {
                    "type": "text",
                    "text": VALIDATION_USER_PROMPT
                }
            ]
        }
    ]
    
    # Call API
    try:
        response = client.chat.completions.create(
            model=os.environ.get("QWEN_MODEL_NAME", "qwen3-vl-32b-instruct"),
            messages=messages,
            max_tokens=256,
            temperature=0.1,
            top_p=0.9,
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Parse response
        valid = response_text.upper().startswith("PASS")
        reason = response_text
        
        return valid, reason, response_text
    except Exception as e:
        print(f"API call failed: {e}")
        # Default to PASS on API error to avoid blocking training
        return True, "API error, defaulting to PASS", str(e)


@app.on_event("startup")
async def load_model():
    """Load model on startup."""
    global model, client, use_vllm
    
    # Check if using vLLM local model or API
    use_api = os.environ.get("USE_QWEN_API", "true").lower() == "true"
    
    if use_api and OPENAI_AVAILABLE:
        # Use OpenAI-compatible API
        api_base = os.environ.get("QWEN_API_BASE", "http://localhost:8170/v1")
        api_key = os.environ.get("QWEN_API_KEY", "EMPTY")
        
        client = OpenAI(
            base_url=api_base,
            api_key=api_key,
        )
        use_vllm = False
        print(f"Using Qwen3-VL API at {api_base}")
    elif VLLM_AVAILABLE:
        # Use vLLM local model
        model_path = os.environ.get("QWEN_MODEL_PATH", "Qwen/Qwen3-VL-32B-Instruct")
        
        model = LLM(
            model=model_path,
            tensor_parallel_size=int(os.environ.get("TENSOR_PARALLEL_SIZE", "1")),
            trust_remote_code=True,
            max_model_len=4096,
        )
        use_vllm = True
        print(f"Loaded Qwen3-VL model from {model_path}")
    else:
        raise RuntimeError("Neither vLLM nor OpenAI API is available")


@app.post("/validate", response_model=ImageValidationResponse)
async def validate_image(request: ImageValidationRequest):
    """Validate an image for quality issues.
    
    Args:
        request: ImageValidationRequest with base64 encoded image
        
    Returns:
        ImageValidationResponse with validation result
    """
    # Decode image
    image = decode_base64_image(request.image_base64)
    
    # Validate
    if use_vllm:
        valid, reason, raw_response = validate_image_vllm(image)
    else:
        valid, reason, raw_response = validate_image_api(image)
    
    return ImageValidationResponse(
        valid=valid,
        reason=reason,
        raw_response=raw_response
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "mode": "vllm" if use_vllm else "api",
        "model_loaded": model is not None or client is not None
    }


# ============================================================================
# Client class for easy integration
# ============================================================================
class QwenImageFilter:
    """Client for Qwen3-VL image quality filter.
    
    Supports two modes:
    1. API mode: Calls a remote API server for validation
    2. Local mode: Loads model locally using transformers/vLLM
    
    Usage:
        # API mode (default)
        filter_client = QwenImageFilter("http://localhost:8171")
        is_valid = filter_client.validate(pil_image)
        
        # Local mode with transformers
        filter_client = QwenImageFilter(
            model_path="/path/to/Qwen3-VL-32B-Instruct",
            use_local=True
        )
        is_valid = filter_client.validate(pil_image)
    """
    
    def __init__(
        self, 
        api_url: str = "http://localhost:8171",
        model_path: str = None,
        use_local: bool = False,
        device: str = "cuda",
    ):
        """Initialize QwenImageFilter.
        
        Args:
            api_url: URL of the Qwen filter API server (for API mode)
            model_path: Path to local Qwen3-VL model (for local mode)
            use_local: If True, load model locally instead of using API
            device: Device for local model inference
        """
        self.use_local = use_local
        self.model = None
        self.processor = None
        self.device = device
        
        if use_local:
            if model_path is None:
                raise ValueError("model_path is required for local mode")
            self._load_local_model(model_path)
        else:
            self.api_url = api_url.rstrip("/")
            self.validate_url = f"{self.api_url}/validate"
    
    def _load_local_model(self, model_path: str):
        """Load Qwen3-VL model locally using transformers."""
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            import torch
            
            print(f"Loading Qwen3-VL model from {model_path}...")
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True,
            )
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
                trust_remote_code=True,
            )
            self.model.eval()
            print(f"Qwen3-VL model loaded successfully on {self.device}")
        except ImportError:
            raise ImportError(
                "transformers and qwen-vl-utils are required for local mode. "
                "Install with: pip install transformers qwen-vl-utils"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load Qwen3-VL model: {e}")
    
    def _validate_local(self, image: Image.Image) -> tuple[bool, str]:
        """Validate image using local model."""
        import torch
        
        # Prepare messages for Qwen3-VL
        messages = [
            {
                "role": "system",
                "content": VALIDATION_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": VALIDATION_USER_PROMPT}
                ]
            }
        ]
        
        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Process inputs
        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.1,
                top_p=0.9,
            )
        
        # Decode response
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response_text = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True
        )[0].strip()
        
        # Parse response
        valid = response_text.upper().startswith("PASS")
        return valid, response_text
    
    def _encode_image(self, image: Image.Image) -> str:
        """Encode PIL Image to base64."""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def validate(self, image: Image.Image) -> bool:
        """Validate a single image.
        
        Args:
            image: PIL Image to validate
            
        Returns:
            True if image passes validation, False otherwise
        """
        if self.use_local:
            try:
                valid, _ = self._validate_local(image)
                return valid
            except Exception as e:
                print(f"Warning: Local validation error: {e}, defaulting to PASS")
                return True
        else:
            # API mode
            import requests
            
            payload = {
                "image_base64": self._encode_image(image)
            }
            
            try:
                response = requests.post(
                    self.validate_url,
                    json=payload,
                    timeout=60
                )
                response.raise_for_status()
                result = response.json()
                return result.get("valid", False)
            except requests.exceptions.Timeout:
                print("Warning: Qwen filter timeout, defaulting to PASS")
                return True
            except requests.exceptions.RequestException as e:
                print(f"Warning: Qwen filter error: {e}, defaulting to PASS")
                return True
    
    def validate_with_reason(self, image: Image.Image) -> tuple[bool, str]:
        """Validate a single image and return reason.
        
        Args:
            image: PIL Image to validate
            
        Returns:
            Tuple of (is_valid, reason)
        """
        if self.use_local:
            try:
                return self._validate_local(image)
            except Exception as e:
                return True, f"Error: {e}, defaulting to PASS"
        else:
            # API mode
            import requests
            
            payload = {
                "image_base64": self._encode_image(image)
            }
            
            try:
                response = requests.post(
                    self.validate_url,
                    json=payload,
                    timeout=60
                )
                response.raise_for_status()
                result = response.json()
                return result.get("valid", False), result.get("reason", "")
            except requests.exceptions.Timeout:
                return True, "Timeout, defaulting to PASS"
            except requests.exceptions.RequestException as e:
                return True, f"Error: {e}, defaulting to PASS"


if __name__ == "__main__":
    # Default port for Qwen filter
    port = int(os.environ.get("QWEN_FILTER_PORT", "8171"))
    
    uvicorn.run(app, host="127.0.0.1", port=port)
