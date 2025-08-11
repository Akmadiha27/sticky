import asyncio
from typing import Annotated
import os
from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
from mcp import ErrorData, McpError
from mcp.server.auth.provider import AccessToken
from mcp.types import TextContent, ImageContent, INVALID_PARAMS, INTERNAL_ERROR
from pydantic import BaseModel, Field, AnyUrl

import markdownify
import httpx
import readabilipy
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import io
import base64
import google.generativeai as genai
import textwrap
from bs4 import BeautifulSoup
import numpy as np
import random
import math

# --- Load environment variables ---
load_dotenv()

TOKEN = os.environ.get("AUTH_TOKEN")
MY_NUMBER = os.environ.get("MY_NUMBER")
REMOVE_BG_API_KEY = os.environ.get("REMOVE_BG_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
PORT = int(os.environ.get("PORT", 8086))  # Render compatibility

assert TOKEN is not None, "Please set AUTH_TOKEN in your .env file"
assert MY_NUMBER is not None, "Please set MY_NUMBER in your .env file"
assert REMOVE_BG_API_KEY is not None, "Please set REMOVE_BG_API_KEY in your .env file"
assert GEMINI_API_KEY is not None, "Please set GEMINI_API_KEY in your .env file"

# Initialize Gemini
genai.configure(api_key=GEMINI_API_KEY)

# --- Auth Provider ---
class SimpleBearerAuthProvider(BearerAuthProvider):
    def __init__(self, token: str):
        k = RSAKeyPair.generate()
        super().__init__(public_key=k.public_key, jwks_uri=None, issuer=None, audience=None)
        self.token = token

    async def load_access_token(self, token: str) -> AccessToken | None:
        if token == self.token:
            return AccessToken(
                token=token,
                client_id="puch-client",
                scopes=["*"],
                expires_at=None,
            )
        return None

# --- Rich Tool Description model ---
class RichToolDescription(BaseModel):
    description: str
    use_when: str
    side_effects: str | None = None

# --- Font Helper ---
def get_font(font_size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Get the best available font for the system with modern sticker styling"""
    if bold:
        font_paths = [
            # Bold/Black fonts for sticker text
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
            "/System/Library/Fonts/Arial Black.ttf",
            "/System/Library/Fonts/Impact.ttf",
            "arialbd.ttf",
            "impact.ttf",
        ]
    else:
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
            "arial.ttf",
            "/System/Library/Fonts/Arial.ttf",
        ]
    
    for font_path in font_paths:
        try:
            return ImageFont.truetype(font_path, font_size)
        except (OSError, IOError):
            continue
    
    # Fallback to default font
    return ImageFont.load_default()

# --- Art Style Configurations ---
ART_STYLES = {
    "ghibli": {
        "saturation_boost": 1.3,
        "contrast_boost": 1.1,
        "blur_radius": 1,
        "sharpness": 1.5,
        "warm_tint": (255, 240, 200, 20),
        "description": "Studio Ghibli anime style with soft, dreamy colors"
    },
    "comic": {
        "saturation_boost": 1.5,
        "contrast_boost": 1.3,
        "blur_radius": 0.5,
        "sharpness": 2.0,
        "warm_tint": (255, 255, 255, 10),
        "description": "Bold comic book style with high contrast"
    },
    "watercolor": {
        "saturation_boost": 1.1,
        "contrast_boost": 0.9,
        "blur_radius": 2,
        "sharpness": 0.8,
        "warm_tint": (250, 245, 240, 30),
        "description": "Soft watercolor painting style"
    }
}

# --- Enhanced Cartoon Effect Functions ---
def apply_cartoon_effect(image, style="ghibli"):
    """Apply customizable cartoon-like effects based on art style"""
    try:
        # Convert to RGB if not already
        if image.mode != 'RGB':
            rgb_image = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'RGBA':
                rgb_image.paste(image, mask=image.split()[-1])
            else:
                rgb_image.paste(image)
            image = rgb_image

        style_config = ART_STYLES.get(style, ART_STYLES["ghibli"])
        
        # Enhance saturation for more vibrant colors
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(style_config["saturation_boost"])
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(style_config["contrast_boost"])
        
        # Apply style-specific smoothing
        if style_config["blur_radius"] > 0:
            image = image.filter(ImageFilter.GaussianBlur(radius=style_config["blur_radius"]))
        
        # Apply sharpening for cartoon effect
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(style_config["sharpness"])
        
        # Additional style-specific effects
        if style == "comic":
            # Add slight edge enhancement for comic style
            image = image.filter(ImageFilter.EDGE_ENHANCE)
        elif style == "watercolor":
            # Add soft blur for watercolor effect
            image = image.filter(ImageFilter.SMOOTH_MORE)
        
        return image
        
    except Exception as e:
        print(f"Cartoon effect failed: {e}")
        return image  # Return original if processing fails

def add_modern_sticker_text(image, text, style="bold", position="smart", text_color=None):
    """Add modern messaging sticker-style text with creative positioning"""
    if not text.strip():
        return image
    
    draw = ImageDraw.Draw(image)
    width, height = image.size
    
    # Dynamic font sizing based on image size and text length
    text_length = len(text)
    base_font_size = max(24, min(width, height) // 15)
    
    if text_length <= 8:
        font_size = int(base_font_size * 1.4)
    elif text_length <= 15:
        font_size = int(base_font_size * 1.2)
    elif text_length <= 25:
        font_size = base_font_size
    else:
        font_size = int(base_font_size * 0.8)
    
    # Get bold font for sticker effect
    font = get_font(font_size, bold=True)
    
    # Wrap text intelligently
    if text_length > 20:
        wrapped_text = textwrap.fill(text, width=20)
    else:
        wrapped_text = text
    
    # Calculate text dimensions
    try:
        bbox = draw.textbbox((0, 0), wrapped_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    except AttributeError:
        text_width, text_height = draw.textsize(wrapped_text, font=font)
    
    # Smart positioning system
    positions = {
        "center": (width // 2 - text_width // 2, height // 2 - text_height // 2),
        "bottom": (width // 2 - text_width // 2, height - text_height - 30),
        "top": (width // 2 - text_width // 2, 30),
        "bottom_left": (30, height - text_height - 30),
        "bottom_right": (width - text_width - 30, height - text_height - 30),
        "top_left": (30, 30),
        "top_right": (width - text_width - 30, 30),
    }
    
    if position == "smart":
        # Analyze image to find best text placement
        # For now, default to bottom center but could be enhanced with image analysis
        position = "bottom"
    
    x, y = positions.get(position, positions["center"])
    
    # Ensure text stays within bounds
    x = max(15, min(x, width - text_width - 15))
    y = max(15, min(y, height - text_height - 15))
    
    # Automatic text color selection for contrast
    if text_color is None:
        # Sample background color at text position
        bg_sample = image.crop((x, y, min(x + text_width, width-1), min(y + text_height, height-1)))
        bg_avg = np.array(bg_sample).mean(axis=(0, 1))
        
        # Choose contrasting color
        if bg_avg.mean() > 128:  # Light background
            text_color = (20, 20, 20)  # Dark text
            outline_color = (255, 255, 255)
        else:  # Dark background
            text_color = (255, 255, 255)  # White text
            outline_color = (20, 20, 20)
    else:
        outline_color = (0, 0, 0) if sum(text_color) > 384 else (255, 255, 255)
    
    # Modern sticker text styling
    if style == "bold":
        # Strong outline for readability
        outline_width = max(2, font_size // 12)
        for dx, dy in [(i, j) for i in range(-outline_width, outline_width + 1) 
                       for j in range(-outline_width, outline_width + 1)]:
            if dx != 0 or dy != 0:
                draw.text((x + dx, y + dy), wrapped_text, font=font, fill=outline_color)
        
        # Main text
        draw.text((x, y), wrapped_text, font=font, fill=text_color)
        
    elif style == "bubble":
        # Background bubble
        padding = max(15, font_size // 4)
        bubble_x1 = x - padding
        bubble_y1 = y - padding // 2
        bubble_x2 = x + text_width + padding
        bubble_y2 = y + text_height + padding // 2
        
        # Rounded rectangle background
        corner_radius = min(20, padding)
        draw.rounded_rectangle(
            [bubble_x1, bubble_y1, bubble_x2, bubble_y2],
            radius=corner_radius,
            fill=(255, 255, 255, 220),
            outline=(200, 200, 200, 255),
            width=2
        )
        
        # Text
        draw.text((x, y), wrapped_text, font=font, fill=(50, 50, 50))
        
    elif style == "glow":
        # Multiple glow layers for modern effect
        glow_colors = [(255, 255, 255, 150), (text_color[0], text_color[1], text_color[2], 100)]
        glow_sizes = [4, 2]
        
        for glow_color, glow_size in zip(glow_colors, glow_sizes):
            for dx in range(-glow_size, glow_size + 1):
                for dy in range(-glow_size, glow_size + 1):
                    if dx != 0 or dy != 0:
                        distance = math.sqrt(dx**2 + dy**2)
                        if distance <= glow_size:
                            alpha = int(glow_color[3] * (1 - distance / glow_size))
                            glow = glow_color[:3] + (alpha,)
                            draw.text((x + dx, y + dy), wrapped_text, font=font, fill=glow)
        
        # Main text
        draw.text((x, y), wrapped_text, font=font, fill=text_color)
    
    return image

# --- Gemini Integration Functions ---
async def analyze_image_with_gemini(image_bytes: bytes, user_caption: str = None) -> tuple[str, str]:
    """Use Gemini to analyze image and generate caption"""
    try:
        if user_caption:
            # User provided caption - just return it with a simple scene description
            return user_caption, "magical dreamy background with soft colors"
        
        # Convert image to PIL Image for Gemini
        image = Image.open(io.BytesIO(image_bytes))
        
        # Generate caption using Gemini Vision
        def call_gemini():
            try:
                model = genai.GenerativeModel('gemini-1.5-flash')
                prompt = "Analyze this image and create a short, fun, magical caption for a messaging sticker (max 6 words). Focus on the main subject/character and make it playful and engaging. Just return the caption text, nothing else."
                
                response = model.generate_content([prompt, image])
                return response.text.strip()
            except Exception as e:
                print(f"Gemini analysis error: {e}")
                return "✨ Pure Magic ✨"  # Fallback caption
        
        # Run in executor to avoid blocking
        loop = asyncio.get_running_loop()
        caption = await loop.run_in_executor(None, call_gemini)
        
        # Clean up the caption (remove quotes, extra punctuation)
        caption = caption.strip('"').strip("'").strip()
        
        return caption, "magical dreamy background"
        
    except Exception as e:
        print(f"Gemini image analysis failed: {e}")
        fallback_caption = user_caption or "✨ Pure Magic ✨"
        return fallback_caption, "magical dreamy background"

# --- Enhanced transformation using local effects (since Gemini doesn't do image generation like DALL-E) ---
async def transform_image_with_effects(image_bytes: bytes, art_style: str = "ghibli") -> bytes:
    """Enhanced local image transformation with advanced effects"""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply multiple enhancement passes for better cartoon effect
        style_config = ART_STYLES.get(art_style, ART_STYLES["ghibli"])
        
        # First pass: Basic cartoon effect
        transformed_image = apply_cartoon_effect(image, art_style)
        
        # Second pass: Style-specific enhancements
        if art_style == "ghibli":
            # Add soft anime-like glow
            enhancer = ImageEnhance.Brightness(transformed_image)
            transformed_image = enhancer.enhance(1.05)
            
            # Slight warm tint
            overlay = Image.new('RGB', transformed_image.size, (255, 245, 220))
            transformed_image = Image.blend(transformed_image, overlay, 0.1)
            
        elif art_style == "comic":
            # High contrast comic effect
            enhancer = ImageEnhance.Contrast(transformed_image)
            transformed_image = enhancer.enhance(1.4)
            
            # Edge enhancement for comic book look
            transformed_image = transformed_image.filter(ImageFilter.EDGE_ENHANCE_MORE)
            
        elif art_style == "watercolor":
            # Watercolor blur effect
            transformed_image = transformed_image.filter(ImageFilter.GaussianBlur(radius=1.5))
            
            # Soft color blending
            enhancer = ImageEnhance.Color(transformed_image)
            transformed_image = enhancer.enhance(0.9)
        
        # Convert back to bytes
        buf = io.BytesIO()
        transformed_image.save(buf, format="PNG")
        return buf.getvalue()
        
    except Exception as e:
        print(f"Image transformation failed: {e}")
        # Return original if transformation fails
        return image_bytes

# --- Background Processing (keeping original logic) ---
def create_simple_background(width, height, style="ghibli"):
    """Create a simple background that will be removed anyway"""
    # Since background gets removed, just create a simple gradient
    colors = {
        "ghibli": [(135, 206, 250), (255, 182, 193)],  # Sky blue to pink
        "comic": [(255, 255, 255), (240, 240, 240)],   # White gradient
        "watercolor": [(255, 248, 220), (255, 240, 245)]  # Cream to light pink
    }
    
    color_pair = colors.get(style, colors["ghibli"])
    background = Image.new('RGBA', (width, height), color_pair[0])
    
    # Simple gradient
    draw = ImageDraw.Draw(background)
    for y in range(height):
        progress = y / height
        color = tuple(int(color_pair[0][i] * (1-progress) + color_pair[1][i] * progress) for i in range(3))
        draw.line([(0, y), (width, y)], fill=color + (255,))
    
    return background

# --- Fetch Utility Class (keeping original) ---
class Fetch:
    USER_AGENT = "Puch/1.0 (Autonomous)"

    @classmethod
    async def fetch_url(
        cls,
        url: str,
        user_agent: str,
        force_raw: bool = False,
    ) -> tuple[str, str]:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    url,
                    follow_redirects=True,
                    headers={"User-Agent": user_agent},
                    timeout=30,
                )
            except httpx.HTTPError as e:
                raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch {url}: {e!r}"))

            if response.status_code >= 400:
                raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch {url} - status code {response.status_code}"))

            page_raw = response.text

        content_type = response.headers.get("content-type", "")
        is_page_html = "text/html" in content_type

        if is_page_html and not force_raw:
            return cls.extract_content_from_html(page_raw), ""

        return (
            page_raw,
            f"Content type {content_type} cannot be simplified to markdown, but here is the raw content:\n",
        )

    @staticmethod
    def extract_content_from_html(html: str) -> str:
        """Extract and convert HTML content to Markdown format."""
        try:
            ret = readabilipy.simple_json.simple_json_from_html_string(html, use_readability=True)
            if not ret or not ret.get("content"):
                return "<error>Page failed to be simplified from HTML</error>"
            content = markdownify.markdownify(ret["content"], heading_style=markdownify.ATX)
            return content
        except Exception as e:
            return f"<error>Failed to parse HTML: {e}</error>"

    @staticmethod
    async def google_search_links(query: str, num_results: int = 5) -> list[str]:
        ddg_url = f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}"
        links = []

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(ddg_url, headers={"User-Agent": Fetch.USER_AGENT}, timeout=10)
                if resp.status_code != 200:
                    return ["<error>Failed to perform search.</error>"]

            soup = BeautifulSoup(resp.text, "html.parser")
            for a in soup.find_all("a", class_="result__a", href=True):
                href = a["href"]
                if "http" in href:
                    links.append(href)
                if len(links) >= num_results:
                    break

            return links or ["<error>No results found.</error>"]
        except Exception as e:
            return [f"<error>Search failed: {e}</error>"]

# --- MCP Server Setup ---
mcp = FastMCP(
    "Gemini Smart Sticker MCP Server",
    auth=SimpleBearerAuthProvider(TOKEN),
)

# --- Tool: validate (required by Puch) ---
@mcp.tool
async def validate() -> str:
    return MY_NUMBER

# --- Tool: job_finder (keeping original) ---
JobFinderDescription = RichToolDescription(
    description="Smart job tool: analyze descriptions, fetch URLs, or search jobs based on free text.",
    use_when="Use this to evaluate job descriptions or search for jobs using freeform goals.",
    side_effects="Returns insights, fetched job descriptions, or relevant job links.",
)

@mcp.tool(description=JobFinderDescription.model_dump_json())
async def job_finder(
    user_goal: Annotated[str, Field(description="The user's goal (can be a description, intent, or freeform query)")],
    job_description: Annotated[str | None, Field(description="Full job description text, if available.")] = None,
    job_url: Annotated[AnyUrl | None, Field(description="A URL to fetch a job description from.")] = None,
    raw: Annotated[bool, Field(description="Return raw HTML content if True")] = False,
) -> str:
    if job_description:
        return (
            f"📝 **Job Description Analysis**\n\n"
            f"---\n{job_description.strip()}\n---\n\n"
            f"User Goal: **{user_goal}**\n\n"
            f"💡 Suggestions:\n- Tailor your resume.\n- Evaluate skill match.\n- Consider applying if relevant."
        )

    if job_url:
        try:
            content, _ = await Fetch.fetch_url(str(job_url), Fetch.USER_AGENT, force_raw=raw)
            return (
                f"🔗 **Fetched Job Posting from URL**: {job_url}\n\n"
                f"---\n{content.strip()}\n---\n\n"
                f"User Goal: **{user_goal}**"
            )
        except McpError:
            raise
        except Exception as e:
            raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch job URL: {e}"))

    if "look for" in user_goal.lower() or "find" in user_goal.lower():
        links = await Fetch.google_search_links(user_goal)
        return (
            f"🔍 **Search Results for**: _{user_goal}_\n\n" +
            "\n".join(f"- {link}" for link in links)
        )

    raise McpError(ErrorData(code=INVALID_PARAMS, message="Please provide either a job description, a job URL, or a search query in user_goal."))

# --- Tool: make_img_black_and_white (keeping original) ---
MAKE_IMG_BLACK_AND_WHITE_DESCRIPTION = RichToolDescription(
    description="Convert an image to black and white and save it.",
    use_when="Use this tool when the user provides an image URL and requests it to be converted to black and white.",
    side_effects="The image will be processed and saved in a black and white format.",
)

@mcp.tool(description=MAKE_IMG_BLACK_AND_WHITE_DESCRIPTION.model_dump_json())
async def make_img_black_and_white(
    puch_image_data: Annotated[str, Field(description="Base64-encoded image data to convert to black and white")] = None,
) -> list[TextContent | ImageContent]:
    try:
        if not puch_image_data:
            raise McpError(ErrorData(code=INVALID_PARAMS, message="puch_image_data is required"))
            
        # Check for placeholder/invalid data
        if puch_image_data in ['Base64 Image', 'base64_image_data', '', 'undefined', 'null']:
            raise McpError(ErrorData(
                code=INVALID_PARAMS, 
                message="❌ PuchAI integration error: Image not converted to base64. This is a PuchAI bug - it should convert images to base64 before sending to MCP server."
            ))
        
        image_bytes = base64.b64decode(puch_image_data)
        image = Image.open(io.BytesIO(image_bytes))

        bw_image = image.convert("L")

        buf = io.BytesIO()
        bw_image.save(buf, format="PNG")
        bw_bytes = buf.getvalue()
        bw_base64 = base64.b64encode(bw_bytes).decode("utf-8")

        return [ImageContent(type="image", mimeType="image/png", data=bw_base64)]
    except Exception as e:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Black and white conversion failed: {e}"))

# --- Tool: remove_bg (keeping original) ---
async def remove_bg(image_bytes: bytes, api_key: str) -> bytes:
    url = "https://api.remove.bg/v1.0/removebg"
    headers = {"X-Api-Key": api_key}
    files = {"image_file": ("image.png", image_bytes)}
    
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, headers=headers, files=files, timeout=60)
            if resp.status_code != 200:
                raise Exception(f"remove.bg API failed: {resp.text}")
            return resp.content
    except httpx.TimeoutException:
        raise Exception("remove.bg API timeout")
    except Exception as e:
        raise Exception(f"remove.bg API error: {e}")

# --- Debug tool ---
@mcp.tool(description="Debug what data PuchAI is actually sending")
async def debug_puch_data(
    puch_image_data: Annotated[str, Field(description="The data PuchAI sends")] = "nothing_received"
) -> str:
    """Debug what PuchAI actually sends"""
    return f"""
🔍 DEBUG REPORT:
- Data received: '{puch_image_data}'
- Data type: {type(puch_image_data)}
- Data length: {len(puch_image_data)} chars
- First 100 chars: {puch_image_data[:100]}
- Is placeholder?: {'YES' if puch_image_data in ['Base64 Image', 'base64_image_data', ''] else 'NO'}

💡 Expected: Long base64 string starting with something like '/9j/4AAQ...' or 'iVBORw0KGgo...'
❌ Problem: PuchAI is not converting images to base64 properly
"""

# --- Enhanced Smart Sticker Maker with Gemini ---
@mcp.tool(description="Create modern messaging stickers with Gemini-powered analysis. AI can generate captions or enhance user-provided ones to better match the image content.")
async def smart_sticker_maker(
    puch_image_data: Annotated[str, Field(description="Base64 image data")],
    caption: Annotated[str | None, Field(description="Optional caption - AI will analyze if it fits the image and may enhance it. If not provided, AI generates one automatically.")] = None,
    art_style: Annotated[str, Field(description="Art style: 'ghibli' (Studio Ghibli anime), 'comic' (comic book), or 'watercolor' (soft painting)")] = "ghibli",
    text_style: Annotated[str, Field(description="Text style: 'bold' (strong outline), 'bubble' (background bubble), or 'glow' (glowing effect)")] = "bold",
    text_position: Annotated[str, Field(description="Text position: 'smart', 'center', 'bottom', 'top', 'bottom_left', 'bottom_right', 'top_left', 'top_right'")] = "smart",
    use_enhanced_transform: Annotated[bool, Field(description="Use enhanced local transformation effects (recommended for better quality)")] = True,
) -> list[ImageContent]:
    if not REMOVE_BG_API_KEY:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message="Remove.bg API key not configured."))

    try:
        # Step 0: Validate the image
        if not puch_image_data:
            raise McpError(ErrorData(code=INVALID_PARAMS, message="puch_image_data is required"))
            
        # Check for placeholder/invalid data
        if puch_image_data in ['Base64 Image', 'base64_image_data', '', 'undefined', 'null']:
            raise McpError(ErrorData(
                code=INVALID_PARAMS, 
                message="❌ PuchAI integration error: Image not converted to base64. This is a PuchAI bug - it should convert images to base64 before sending to MCP server."
            ))
        
        try:
            # Remove data URL prefix if present
            if puch_image_data.startswith('data:'):
                puch_image_data = puch_image_data.split(',', 1)[1]
            
            image_bytes = base64.b64decode(puch_image_data, validate=True)
            image = Image.open(io.BytesIO(image_bytes))
            image.verify()
            image = Image.open(io.BytesIO(image_bytes))  # Reopen for actual use
        except Exception as e:
            raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Invalid image data: {e}"))

        # Step 1: Generate AI caption using Gemini
        try:
            ai_caption, _ = await analyze_image_with_gemini(image_bytes, caption)
            print(f"🤖 Gemini generated caption: {ai_caption}")
        except Exception as e:
            print(f"Gemini caption generation failed: {e}")
            ai_caption = caption or "✨ Amazing ✨"

        # Step 2: Apply enhanced transformation to CHARACTER ONLY
        if use_enhanced_transform:
            try:
                # Use enhanced local transformation (since Gemini doesn't do image generation)
                print(f"🎨 Using enhanced transformation for {art_style} style...")
                transformed_bytes = await transform_image_with_effects(image_bytes, art_style)
                transformed_image = Image.open(io.BytesIO(transformed_bytes))
                print("✅ Enhanced transformation successful!")
            except Exception as e:
                print(f"❌ Enhanced transformation failed, using basic: {e}")
                # Fallback to basic transformation
                transformed_image = apply_cartoon_effect(image.convert('RGB'), art_style)
        else:
            # Use basic cartoon effect (faster)
            print(f"🎨 Using basic transformation for {art_style} style...")
            transformed_image = apply_cartoon_effect(image.convert('RGB'), art_style)

        # Step 3: Remove background from transformed CHARACTER
        # Convert transformed image back to bytes for remove.bg
        buf = io.BytesIO()
        transformed_image.save(buf, format="PNG")
        transformed_bytes = buf.getvalue()
        
        try:
            print("🔄 Removing background from transformed character...")
            transparent_png_bytes = await remove_bg(transformed_bytes, REMOVE_BG_API_KEY)
            print("✅ Background removal successful!")
        except Exception as e:
            raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Background removal failed: {e}"))

        # Step 4: Load the final subject image (NO BACKGROUND ENHANCEMENT!)
        subject_image = Image.open(io.BytesIO(transparent_png_bytes)).convert("RGBA")
        
        # Step 5: Create MINIMAL canvas (no fancy background since it gets removed anyway)
        canvas_size = (max(512, subject_image.width + 100), max(512, subject_image.height + 100))
        # Simple transparent background - NO ENHANCEMENT!
        background = Image.new('RGBA', canvas_size, (255, 255, 255, 0))  # Transparent
        
        # Step 6: Place character on minimal canvas
        x_offset = (canvas_size[0] - subject_image.width) // 2
        y_offset = (canvas_size[1] - subject_image.height) // 2
        
        final_image = background.copy()
        final_image.paste(subject_image, (x_offset, y_offset), subject_image)
        
        # Step 7: Add modern sticker text
        final_image = add_modern_sticker_text(final_image, ai_caption, text_style, text_position)

        # Step 8: MINIMAL final enhancement (NO background effects!)
        # Only enhance the character slightly, no background processing
        if art_style == "comic":
            # Just a tiny contrast boost for comic style
            enhancer = ImageEnhance.Contrast(final_image)
            final_image = enhancer.enhance(1.02)
        # NO other enhancements to avoid background effects!

        # Step 9: Save as PNG (as requested)
        buf_png = io.BytesIO()
        final_image.save(buf_png, format="PNG", quality=95, optimize=True)
        png_bytes = buf_png.getvalue()
        png_b64 = base64.b64encode(png_bytes).decode("utf-8")

        return [
            ImageContent(type="image", mimeType="image/png", data=png_b64)
        ]

    except McpError:
        raise
    except Exception as e:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Smart sticker maker failed: {e}"))


# --- Additional Tool: list_art_styles ---
@mcp.tool(description="List available art styles for the smart sticker maker")
async def list_art_styles() -> str:
    """List all available art styles with descriptions"""
    result = "🎨 **Available Art Styles:**\n\n"
    
    for style_name, config in ART_STYLES.items():
        result += f"**{style_name.title()}**: {config['description']}\n"
    
    result += "\n📝 **Text Styles:** bold, bubble, glow\n"
    result += "📍 **Text Positions:** smart, center, bottom, top, bottom_left, bottom_right, top_left, top_right"
    
    return result


# --- Run MCP Server ---
async def main():
    print(f"🚀 Starting Gemini-powered MCP server on http://0.0.0.0:{PORT}")
    print(f"🎨 Available art styles: {', '.join(ART_STYLES.keys())}")
    print(f"🤖 Using Gemini API for image analysis and caption generation")
    await mcp.run_async("streamable-http", host="0.0.0.0", port=PORT)

if __name__ == "__main__":
    asyncio.run(main())