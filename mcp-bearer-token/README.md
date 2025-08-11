# Job Finder MCP Server

A FastMCP server that provides job searching capabilities and image processing tools for creating WhatsApp-style stickers.

## Features

- **Job Finder**: Analyze job descriptions, fetch job postings from URLs, and search for jobs
- **Image Processing**: Convert images to black and white
- **Smart Sticker Maker**: Create WhatsApp-style stickers with background removal and AI-generated captions

## Setup

### Local Development

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and fill in your API keys:
   ```bash
   cp .env.example .env
   ```

4. Run the server:
   ```bash
   python main.py
   ```

### Deploy to Render

1. Fork/clone this repository
2. Connect it to your Render account
3. Set the following environment variables in Render dashboard:
   - `AUTH_TOKEN`: Your secure authentication token
   - `MY_NUMBER`: Your identification number
   - `REMOVE_BG_API_KEY`: API key from [remove.bg](https://www.remove.bg/api)
   - `GEMINI_API_KEY`: API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

4. Deploy using the provided `render.yaml` configuration

## API Keys Required

- **Remove.bg API**: For background removal in sticker creation
- **Google Gemini API**: For AI-generated captions

## Tools Available

### 1. `validate`
- Returns your configured number for validation

### 2. `job_finder`
- Analyze job descriptions
- Fetch job postings from URLs
- Search for jobs using natural language queries

### 3. `make_img_black_and_white`
- Convert images to black and white format
- Input: Base64-encoded image data
- Output: PNG format image

### 4. `smart_sticker_maker`
- Create WhatsApp-style stickers
- Features:
  - Background removal
  - AI-generated captions (or custom captions)
  - Text overlay with outline
  - Output in both PNG and WEBP formats

## Usage Example

The server runs on port 8086 (or the PORT environment variable if set) and accepts MCP protocol requests with bearer token authentication.

## Error Handling

The server includes comprehensive error handling for:
- Invalid image data
- API failures (remove.bg, Gemini)
- Network timeouts
- Missing dependencies

## Dependencies

See `requirements.txt` for the complete list of Python dependencies.

## License

MIT License