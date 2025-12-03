# utils/pdf_processor.py

import logging
from pathlib import Path
from typing import Optional
import PyPDF2
import pdfplumber

from config.settings import settings

logger = logging.getLogger("exam_evaluator.pdf_processor")


class PDFProcessor:
    """
    Handles PDF file processing and text extraction.
    Uses configuration from settings.py for validation.
    """
    
    @staticmethod
    def validate_pdf(file_path: Path) -> bool:
        """
        Validate PDF file size and format.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            bool: True if valid
            
        Raises:
            ValueError: If validation fails
        """
        logger.info(f"Validating PDF: {file_path}")
        
        # Check if file exists
        if not file_path.exists():
            raise ValueError(f"PDF file not found: {file_path}")
        
        # Check file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > settings.MAX_PDF_SIZE_MB:
            raise ValueError(
                f"PDF file too large: {file_size_mb:.2f}MB "
                f"(max: {settings.MAX_PDF_SIZE_MB}MB)"
            )
        
        # Check if it's a valid PDF
        try:
            with open(file_path, 'rb') as f:
                PyPDF2.PdfReader(f)
        except Exception as e:
            raise ValueError(f"Invalid PDF file: {str(e)}")
        
        logger.info(f"PDF validation passed", extra={
            "file_path": str(file_path),
            "file_size_mb": round(file_size_mb, 2)
        })
        
        return True
    
    
    @staticmethod
    def extract_text_pypdf2(file_path: Path) -> str:
        """
        Extract text from PDF using PyPDF2.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            str: Extracted text
        """
        logger.info(f"Extracting text with PyPDF2: {file_path}")
        
        try:
            text = ""
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                num_pages = len(pdf_reader.pages)
                
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text()
            
            logger.info(f"Text extracted successfully", extra={
                "num_pages": num_pages,
                "text_length": len(text)
            })
            
            return text
            
        except Exception as e:
            logger.error(f"PyPDF2 extraction failed: {str(e)}", exc_info=True)
            raise
    
    
    @staticmethod
    def extract_text_pdfplumber(file_path: Path) -> str:
        """
        Extract text from PDF using pdfplumber (better for complex layouts).
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            str: Extracted text
        """
        logger.info(f"Extracting text with pdfplumber: {file_path}")
        
        try:
            text = ""
            with pdfplumber.open(file_path) as pdf:
                num_pages = len(pdf.pages)
                
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
            
            logger.info(f"Text extracted successfully", extra={
                "num_pages": num_pages,
                "text_length": len(text)
            })
            
            return text
            
        except Exception as e:
            logger.error(f"pdfplumber extraction failed: {str(e)}", exc_info=True)
            raise
    
    
    @staticmethod
    def extract_text(
        file_path: Path, 
        method: str = "pdfplumber"
    ) -> str:
        """
        Extract text from PDF using specified method.
        Falls back to alternative method if first fails.
        
        Args:
            file_path: Path to PDF file
            method: Extraction method ("pypdf2" or "pdfplumber")
            
        Returns:
            str: Extracted text
        """
        # Validate PDF first
        PDFProcessor.validate_pdf(file_path)
        
        logger.info(f"Starting PDF text extraction", extra={
            "file_path": str(file_path),
            "method": method
        })
        
        try:
            if method == "pdfplumber":
                text = PDFProcessor.extract_text_pdfplumber(file_path)
            else:
                text = PDFProcessor.extract_text_pypdf2(file_path)
            
            # Validate extracted text
            if not text or len(text.strip()) < 100:
                logger.warning("Extracted text too short, trying fallback method")
                
                # Try fallback method
                if method == "pdfplumber":
                    text = PDFProcessor.extract_text_pypdf2(file_path)
                else:
                    text = PDFProcessor.extract_text_pdfplumber(file_path)
            
            # Clean text
            text = PDFProcessor.clean_text(text)
            
            logger.info(f"Text extraction completed", extra={
                "final_text_length": len(text),
                "word_count": len(text.split())
            })
            
            return text
            
        except Exception as e:
            logger.error(f"PDF text extraction failed: {str(e)}", exc_info=True)
            raise
    
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean extracted text by removing extra whitespace and special characters.
        
        Args:
            text: Raw extracted text
            
        Returns:
            str: Cleaned text
        """
        # Remove multiple newlines
        text = "\n".join(line.strip() for line in text.split("\n") if line.strip())
        
        # Remove multiple spaces
        import re
        text = re.sub(r' +', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-\']', ' ', text)
        
        return text.strip()
    
    
    @staticmethod
    def save_pdf(
        file_content: bytes, 
        filename: str
    ) -> Path:
        """
        Save uploaded PDF file to storage.
        
        Args:
            file_content: PDF file content as bytes
            filename: Original filename
            
        Returns:
            Path: Path where file was saved
        """
        # Create upload directory if not exists
        upload_dir = Path(settings.UPLOADED_PDFS_PATH)
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate safe filename
        safe_filename = "".join(
            c for c in filename if c.isalnum() or c in "._-"
        )
        
        file_path = upload_dir / safe_filename
        
        # Save file
        with open(file_path, 'wb') as f:
            f.write(file_content)
        
        logger.info(f"PDF saved", extra={
            "file_path": str(file_path),
            "file_size_bytes": len(file_content)
        })
        
        return file_path