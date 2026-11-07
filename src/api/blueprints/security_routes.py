"""
Security and file upload routes
"""
from pathlib import Path
from flask import Blueprint, request, jsonify, current_app

from src.utils.security import SecureFileHandler, rate_limiter, get_client_ip, SecurityError
from src.utils.language_detector import LanguageDetector


def create_security_blueprint(output_dir):
    """
    Create and configure the security blueprint

    Args:
        output_dir: Base directory for file operations
    """
    bp = Blueprint('security', __name__)

    # Initialize secure file handler
    upload_dir = Path(output_dir) / 'uploads'
    secure_file_handler = SecureFileHandler(upload_dir)

    @bp.route('/api/upload', methods=['POST'])
    def upload_file():
        """Secure file upload with comprehensive validation"""

        # Rate limiting
        client_ip = get_client_ip(request)
        if not rate_limiter.is_allowed(client_ip):
            return jsonify({
                "error": "Rate limit exceeded. Please wait before uploading again.",
                "remaining_requests": rate_limiter.get_remaining_requests(client_ip)
            }), 429

        # Check if file is present
        if 'file' not in request.files:
            return jsonify({"error": "No file part in request"}), 400

        file = request.files['file']
        if not file or file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Security check: limit filename length in request
        if len(file.filename) > 255:
            return jsonify({"error": "Filename too long"}), 400

        try:
            # Read file data
            file_data = file.read()

            # Quick size check before validation
            if len(file_data) == 0:
                return jsonify({"error": "Empty file not allowed"}), 400

            # Validate and save file securely
            validation_result = secure_file_handler.validate_and_save_file(
                file_data, file.filename
            )

            if not validation_result.is_valid:
                return jsonify({
                    "error": validation_result.error_message,
                    "details": "File validation failed"
                }), 400

            # Determine file type
            original_filename = file.filename.lower()
            if original_filename.endswith('.epub'):
                file_type = "epub"
            elif original_filename.endswith('.srt'):
                file_type = "srt"
            elif original_filename.endswith('.pgn'):
                file_type = "pgn"
            elif original_filename.endswith('.docx'):
                file_type = "docx"
            elif original_filename.endswith('.pdf'):
                file_type = "pdf"
            else:
                file_type = "txt"

            # Get file info
            file_size = len(file_data)
            secure_path = validation_result.file_path

            # Detect source language from file content
            detected_language, confidence = LanguageDetector.detect_language_from_file(
                file_data, file.filename
            )

            # Extract cover for EPUB files
            thumbnail_filename = None
            if file_type == "epub":
                try:
                    from src.core.epub.cover_extractor import EPUBCoverExtractor

                    # Create thumbnails directory
                    thumbnails_dir = Path(output_dir) / 'thumbnails'
                    thumbnails_dir.mkdir(exist_ok=True)

                    # Extract and save thumbnail
                    thumbnail_filename = EPUBCoverExtractor.extract_cover(
                        str(secure_path),
                        thumbnails_dir
                    )

                    if thumbnail_filename:
                        current_app.logger.info(f"Extracted EPUB cover: {thumbnail_filename}")
                except Exception as e:
                    current_app.logger.warning(f"Failed to extract EPUB cover: {e}")
                    # Continue without thumbnail (graceful degradation)

            # Return success response
            response_data = {
                "success": True,
                "file_path": str(secure_path),
                "filename": file.filename,
                "secure_filename": secure_path.name,
                "file_type": file_type,
                "size": file_size,
                "size_mb": round(file_size / (1024 * 1024), 2)
            }

            # Add thumbnail if available
            if thumbnail_filename:
                response_data["thumbnail"] = thumbnail_filename

            # Add detected language if available
            if detected_language:
                response_data["detected_language"] = detected_language
                response_data["language_confidence"] = round(confidence, 2)
                current_app.logger.info(
                    f"Language detected: {detected_language} "
                    f"(confidence: {confidence:.2f}) for {file.filename}"
                )

            # Add warnings if any
            if validation_result.warnings:
                response_data["warnings"] = validation_result.warnings

            # Log successful upload
            current_app.logger.info(f"Secure file upload successful: {file.filename} -> {secure_path.name}, Size: {file_size} bytes, IP: {client_ip}")

            return jsonify(response_data), 200

        except SecurityError as e:
            current_app.logger.warning(f"Security violation in file upload: {str(e)}, IP: {client_ip}, Filename: {file.filename}")
            return jsonify({
                "error": "Security validation failed",
                "details": str(e)
            }), 403

        except Exception as e:
            current_app.logger.error(f"File upload error: {str(e)}, IP: {client_ip}, Filename: {file.filename}")
            return jsonify({
                "error": "Upload failed due to server error",
                "details": "Please try again or contact support"
            }), 500

    @bp.route('/api/security/cleanup', methods=['POST'])
    def cleanup_old_files():
        """Clean up old uploaded files (admin endpoint)"""
        try:
            # Get max age from request (default 24 hours)
            max_age_hours = request.json.get('max_age_hours', 24) if request.json else 24

            # Validate input
            if not isinstance(max_age_hours, (int, float)) or max_age_hours < 1:
                return jsonify({"error": "Invalid max_age_hours parameter"}), 400

            # Perform cleanup
            secure_file_handler.cleanup_old_files(max_age_hours)

            return jsonify({
                "success": True,
                "message": f"Cleanup completed for files older than {max_age_hours} hours"
            })

        except Exception as e:
            current_app.logger.error(f"Cleanup error: {str(e)}")
            return jsonify({"error": "Cleanup failed"}), 500

    @bp.route('/api/security/info', methods=['GET'])
    def get_security_info():
        """Get security configuration and limits"""
        client_ip = get_client_ip(request)

        return jsonify({
            "file_limits": {
                "max_size_mb": SecureFileHandler.MAX_FILE_SIZE // (1024 * 1024),
                "allowed_extensions": list(SecureFileHandler.ALLOWED_EXTENSIONS),
                "allowed_mime_types": list(SecureFileHandler.ALLOWED_MIME_TYPES)
            },
            "rate_limit": {
                "remaining_requests": rate_limiter.get_remaining_requests(client_ip),
                "window_seconds": rate_limiter._window_seconds,
                "max_requests": rate_limiter._max_requests
            },
            "upload_directory": str(secure_file_handler.upload_dir)
        })

    @bp.route('/api/uploads/verify', methods=['POST'])
    def verify_uploaded_files():
        """Verify which uploaded files still exist on the server"""
        try:
            data = request.json
            if not data or 'file_paths' not in data:
                return jsonify({"error": "No file paths provided"}), 400

            file_paths = data['file_paths']
            if not isinstance(file_paths, list):
                return jsonify({"error": "Invalid file paths list"}), 400

            existing_files = []
            missing_files = []

            for file_path_str in file_paths:
                file_path = Path(file_path_str)
                if file_path.exists():
                    existing_files.append(file_path_str)
                else:
                    missing_files.append(file_path_str)

            return jsonify({
                "existing": existing_files,
                "missing": missing_files
            })

        except Exception as e:
            current_app.logger.error(f"Error verifying uploaded files: {str(e)}")
            return jsonify({"error": "Verification failed", "details": str(e)}), 500

    @bp.route('/api/detect-language', methods=['POST'])
    def detect_language():
        """Detect language from an already uploaded file"""
        try:
            data = request.json
            if not data or 'file_path' not in data:
                return jsonify({"error": "No file path provided"}), 400

            file_path_str = data['file_path']
            file_path = Path(file_path_str)

            # Security: ensure file exists
            if not file_path.exists():
                return jsonify({"error": "File not found"}), 404

            # Security: ensure file is within upload directory
            resolved = file_path.resolve()
            upload_resolved = secure_file_handler.upload_dir.resolve()
            if not str(resolved).startswith(str(upload_resolved)):
                return jsonify({"error": "Access denied"}), 403

            # Read file and detect language
            with open(file_path, 'rb') as f:
                file_data = f.read()

            detected_language, confidence = LanguageDetector.detect_language_from_file(
                file_data, file_path.name
            )

            if detected_language:
                current_app.logger.info(
                    f"Language detected: {detected_language} "
                    f"(confidence: {confidence:.2f}) for {file_path.name}"
                )
                return jsonify({
                    "success": True,
                    "detected_language": detected_language,
                    "language_confidence": round(confidence, 2)
                })
            else:
                return jsonify({
                    "success": False,
                    "error": "Could not detect language"
                }), 200

        except Exception as e:
            current_app.logger.error(f"Language detection error: {str(e)}")
            return jsonify({"error": "Language detection failed"}), 500

    @bp.route('/api/thumbnails/<path:filename>', methods=['GET'])
    def serve_thumbnail(filename):
        """Serve EPUB cover thumbnail with security validation"""
        try:
            from werkzeug.utils import secure_filename
            from flask import send_file

            # Security: prevent path traversal
            safe_filename = secure_filename(filename)
            if safe_filename != filename or '..' in filename:
                return jsonify({"error": "Invalid filename"}), 400

            thumbnails_dir = Path(output_dir) / 'thumbnails'
            thumbnail_path = thumbnails_dir / safe_filename

            # Security: ensure path is within thumbnails directory
            resolved = thumbnail_path.resolve()
            if not str(resolved).startswith(str(thumbnails_dir.resolve())):
                return jsonify({"error": "Access denied"}), 403

            if not thumbnail_path.exists():
                return jsonify({"error": "Thumbnail not found"}), 404

            # Serve with caching headers
            return send_file(
                thumbnail_path,
                mimetype='image/jpeg',
                as_attachment=False,
                max_age=3600  # Cache 1 hour
            )

        except Exception as e:
            current_app.logger.error(f"Error serving thumbnail: {e}")
            return jsonify({"error": "Failed to serve thumbnail"}), 500

    return bp
