# Fabric Defect Detection & Visual Question Answering

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.0-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Supported-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📌 Overview

A complete end-to-end system for **fabric defect detection** and **visual question answering (VQA)**. The system uses:

- **U-Net** with EfficientNet-b0 encoder for semantic segmentation
- **SmolVLM** (HuggingFace) for visual question answering
- **FastAPI** for web interface
- **MinIO** for image storage
- **SQLite/PostgreSQL** for metadata management

## ✨ Features

- 🔍 **Defect Detection**: U-Net segmentation to identify fabric defects
- 💬 **Visual Q&A**: Ask questions about detected defects using SmolVLM
- 📏 **Real Measurements**: Defect sizes in millimeters with position descriptions
- 🗄️ **Storage**: MinIO for images, SQLite/PostgreSQL for metadata
- 🐳 **Docker Support**: Easy deployment with Docker Compose
- 🌐 **Web Interface**: User-friendly UI for image upload and Q&A