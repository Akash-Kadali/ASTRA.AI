#!/bin/bash

# ============================================================
# ASTRA macOS Build Script
# ============================================================
# Run from the ASTRA project directory:
#   chmod +x build_astra.sh
#   ./build_astra.sh
# ============================================================

set -e  # Exit on error

echo "=============================================="
echo "  ASTRA macOS App Builder"
echo "=============================================="
echo ""

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    echo "❌ Error: main.py not found!"
    echo "   Please run this script from the ASTRA project root."
    exit 1
fi

# Check for icon file
if [ ! -f "icon.icns" ]; then
    echo "⚠️  Warning: icon.icns not found!"
    echo "   The app will be built without a custom icon."
    echo "   Press Enter to continue or Ctrl+C to cancel..."
    read
    ICON_FLAG=""
else
    echo "✅ Found icon.icns"
    ICON_FLAG='--icon "icon.icns"'
fi

# Check for pyinstaller
if ! command -v pyinstaller &> /dev/null; then
    echo "❌ Error: PyInstaller not found!"
    echo "   Install it with: pip install pyinstaller"
    exit 1
fi

echo "🧹 Cleaning previous builds..."
rm -rf build dist *.spec

echo "📦 Building ASTRA.app..."
echo ""

# Build command
pyinstaller \
  --noconfirm \
  --clean \
  --onedir \
  --windowed \
  --name "ASTRA" \
  --icon "icon.icns" \
  --osx-bundle-identifier "com.astra.app" \
  "main.py" \
  --exclude-module nltk \
  --add-data "frontend:frontend" \
  --add-data ".env:." \
  --add-data "backend/.env:backend" \
  --add-data "data:data" \
  --add-data "backend/templates:backend/templates" \
  --add-data "backend/api/base_coverletter.tex:backend/api" \
  --add-data "backend/api/base_resume.tex:backend/api" \
  --collect-submodules backend \
  --hidden-import backend.api.optimize \
  --hidden-import backend.api.coverletter \
  --hidden-import backend.api.talk \
  --hidden-import backend.api.superhuman \
  --hidden-import backend.api.humanize \
  --hidden-import backend.api.mastermind \
  --hidden-import backend.api.dashboard \
  --hidden-import backend.api.models_router \
  --hidden-import backend.api.context_store \
  --hidden-import backend.api.utils_router \
  --hidden-import backend.api.debug \
  --hidden-import backend.core.config \
  --hidden-import backend.core.compiler \
  --hidden-import backend.core.utils \
  --hidden-import backend.core.security \
  --hidden-import backend.core.mastermind_memory \
  --hidden-import uvicorn \
  --hidden-import uvicorn.logging \
  --hidden-import uvicorn.loops \
  --hidden-import uvicorn.loops.auto \
  --hidden-import uvicorn.protocols \
  --hidden-import uvicorn.protocols.http \
  --hidden-import uvicorn.protocols.http.auto \
  --hidden-import uvicorn.protocols.websockets \
  --hidden-import uvicorn.protocols.websockets.auto \
  --hidden-import uvicorn.lifespan \
  --hidden-import uvicorn.lifespan.on \
  --hidden-import fastapi \
  --hidden-import starlette \
  --hidden-import dotenv \
  --hidden-import pydantic \
  --hidden-import httptools \
  --hidden-import websockets \
  --hidden-import anyio

echo ""
echo "🎨 Ensuring icon is properly set..."

# Check if app was built
if [ ! -d "dist/ASTRA.app" ]; then
    echo "❌ Error: ASTRA.app was not created!"
    exit 1
fi

# Copy icon if not present in Resources
if [ -f "icon.icns" ] && [ ! -f "dist/ASTRA.app/Contents/Resources/icon.icns" ]; then
    echo "   Copying icon to Resources..."
    cp icon.icns "dist/ASTRA.app/Contents/Resources/icon.icns"
fi

# Update Info.plist to ensure icon is referenced
if [ -f "dist/ASTRA.app/Contents/Info.plist" ]; then
    echo "   Updating Info.plist..."
    /usr/libexec/PlistBuddy -c "Set :CFBundleIconFile icon" "dist/ASTRA.app/Contents/Info.plist" 2>/dev/null || \
    /usr/libexec/PlistBuddy -c "Add :CFBundleIconFile string icon" "dist/ASTRA.app/Contents/Info.plist" 2>/dev/null || true
fi

echo "🔄 Refreshing app timestamp..."
touch "dist/ASTRA.app"

# Remove quarantine attribute (prevents "damaged app" warning)
echo "🔓 Removing quarantine attribute..."
xattr -cr "dist/ASTRA.app" 2>/dev/null || true

echo ""
echo "=============================================="
echo "  ✅ Build Complete!"
echo "=============================================="
echo ""
echo "📍 Location: $(pwd)/dist/ASTRA.app"
echo ""
echo "🚀 To run the app, use one of these methods:"
echo ""
echo "   Method 1: Double-click ASTRA.app in Finder"
echo "   Method 2: open dist/ASTRA.app"
echo "   Method 3: ./dist/ASTRA.app/Contents/MacOS/ASTRA (debug mode)"
echo ""
echo "💡 If the icon doesn't appear:"
echo "   1. Run: killall Finder"
echo "   2. Or restart your Mac"
echo ""
echo "⚠️  If you get 'damaged app' warning:"
echo "   Run: xattr -cr dist/ASTRA.app"
echo ""
