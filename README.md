# Container Emptiness Classifier & Order Automation System

An intelligent system that uses Gemini Vision AI to detect product containers in images, assess their fill levels, and automatically place orders for low-stock items.

## Overview

This project combines computer vision and automated ordering to create a smart inventory management system. It can:

- **Detect Products**: Identify containers, bottles, tubes, and other products in images
- **Assess Fill Levels**: Determine how full each container is using visual analysis
- **Auto-Order**: Automatically place orders via Amazon, Instacart, or other vendors when items are low

## Key Features

- 🤖 **AI-Powered Detection**: Uses Google's Gemini 2.5 Flash for accurate product detection
- 📊 **Fill Level Analysis**: Estimates container fullness with confidence scores
- 🛒 **Multi-Vendor Support**: Automatic ordering via Amazon, Instacart, and Walmart
- 📦 **Smart Inventory Tracking**: Monitors product levels and triggers reorders
- 🎯 **Bounding Box Visualization**: Creates annotated images with detected items

## Project Structure

```
├── segmentation.py              # Main vision AI pipeline
├── detect_features.py           # Feature detection with Gemini API
├── order_system/
│   └── order_system_minimal.py  # Automated ordering system
├── assets/                       # Test images
│   ├── soap.jpg
│   ├── water_bottle1.jpg
│   ├── water_bottle2.webp
│   └── multiple_products.png
├── requirements.txt             # Python dependencies
└── README.md
```

## Installation

### Prerequisites

- Python 3.8+
- Google Gemini API Key ([Get one here](https://makersuite.google.com/app/apikey))

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd emptiness_classifier_order_automation
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up API key**
   
   Create a `.env` file in the project root:
   ```bash
   echo "GEMINI_API_KEY=your_api_key_here" > .env
   ```

## Usage

### 1. Product Detection and Fill Analysis

Run the segmentation pipeline on an image:

```bash
python segmentation.py
```

This will:
- Process images in the current directory
- Detect all product containers
- Calculate fill percentages
- Generate annotated output images

**Example Output:**
```
Processing item 1: shampoo bottle, white, pump
  Box coordinates: (120, 150) to (350, 450)
  Percent full: 45%
  Is low: false
  Confidence: 0.9
```

### 2. Feature Detection Only

For quick testing with a single image:

```bash
python detect_features.py
```

### 3. Order System Demo

Test the automated ordering system:

```bash
python order_system/order_system_minimal.py
```

**Example Features:**
- Automatic order placement with vendor APIs
- Tax and shipping calculation
- Order tracking and cancellation
- Multi-vendor support (Amazon, Instacart, Walmart)

## How It Works

### 1. Vision AI Pipeline

The `segmentation.py` script uses Gemini Vision to:

1. **Preprocess Images**: Resize and optimize for API processing
2. **Detect Objects**: Identify all distinct products in the frame
3. **Analyze Fill Levels**: 
   - Transparent containers: Detect liquid boundaries
   - Opaque containers: Use sight windows or defaults
   - Squeezable containers: Analyze creases and flattening
4. **Generate Output**: JSON metadata + annotated images

### 2. Order Automation

The `order_system_minimal.py` provides:

- **Product Database**: Pre-configured products with vendors
- **Order Placement**: Async order processing with vendor APIs
- **Cost Calculation**: Automatic tax and shipping
- **Order Management**: Tracking, cancellation, and history

## Output Format

Each detected product includes:

```json
{
  "box_2d": [y0, x0, y1, x1],
  "label": "product name and description",
  "percent_full": 0-100,
  "is_low": true/false,
  "confidence": 0.0-1.0
}
```

## Configuration

### Adjusting Detection Thresholds

Edit `segmentation.py` to modify:
- **Low Stock Threshold**: Currently set to `< 25%`
- **Confidence Levels**: Fine-tune accuracy requirements
- **Image Processing**: Adjust resize dimensions

### Adding Products to Order System

Edit `order_system/order_system_minimal.py`:

```python
PRODUCTS = {
    "prod_001": Product(
        "prod_001", 
        "Your Product Name", 
        19.99, 
        "Amazon",
        asin="BXXXXXXXXX"
    ),
}
```

## Testing

Test with sample images:
- `assets/soap.jpg` - Single product
- `assets/water_bottle1.jpg` - Transparent container
- `assets/multiple_products.png` - Multiple items
- `assets/water_bottle2.webp` - Different product

## Troubleshooting

### API Timeout Errors
- Reduce image resolution in `segmentation.py`
- Check API rate limits and quotas
- Ensure stable internet connection

### Detection Issues
- Use well-lit, clear images
- Avoid overlapping products
- Ensure products are clearly visible

### Order Placement Failures
- Verify vendor credentials
- Check product availability
- Ensure delivery addresses are valid

## Dependencies

Key packages:
- `google-generativeai` - Gemini Vision API
- `pillow` - Image processing
- `numpy` - Array operations
- `python-dotenv` - Environment variables

See `requirements.txt` for complete list.

## Future Enhancements

- [ ] Real-time camera integration
- [ ] Multi-language support
- [ ] Mobile app interface
- [ ] Advanced inventory analytics
- [ ] Vendor API integrations
- [ ] Automated schedule ordering
- [ ] Barcode/QR code recognition

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Support

For issues or questions, please open an issue on the repository.

---

**Built with**: Python, Google Gemini Vision AI, Pillow, and lots of ☕
