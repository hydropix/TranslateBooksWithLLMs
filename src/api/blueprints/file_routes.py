"""
File management routes (list, download, delete, batch operations)
"""
import io
import zipfile
from datetime import datetime
from flask import Blueprint, request, jsonify, send_from_directory, send_file, current_app

from src.api.services import FileService, PathValidator


def create_file_blueprint(output_dir):
    """
    Create and configure the file management blueprint

    Args:
        output_dir: Base directory for file operations
    """
    bp = Blueprint('files', __name__)
    file_service = FileService(output_dir)

    @bp.route('/api/files', methods=['GET'])
    def list_all_files():
        """List all files in the translated_files directory with metadata"""
        try:
            files_info = file_service.list_all_files()
            total_bytes, total_mb = file_service.get_total_size(files_info)

            return jsonify({
                "files": files_info,
                "total_files": len(files_info),
                "total_size_bytes": total_bytes,
                "total_size_mb": total_mb
            })

        except Exception as e:
            current_app.logger.error(f"Error listing files: {str(e)}")
            return jsonify({"error": "Failed to list files", "details": str(e)}), 500

    @bp.route('/api/files/<path:filename>', methods=['GET'])
    def download_file_by_name(filename):
        """Download a specific file by name"""
        try:
            # Validate filename
            is_valid, error = PathValidator.validate_filename(filename)
            if not is_valid:
                return jsonify({"error": error}), 400

            # Find file
            file_path = file_service.find_file(filename)
            if not file_path:
                return jsonify({"error": "File not found"}), 404

            # Determine directory for send_from_directory
            if file_path.parent == file_service.uploads_dir:
                return send_from_directory(str(file_service.uploads_dir), filename, as_attachment=True)
            else:
                return send_from_directory(output_dir, filename, as_attachment=True)

        except Exception as e:
            current_app.logger.error(f"Error downloading file {filename}: {str(e)}")
            return jsonify({"error": "Download failed", "details": str(e)}), 500

    @bp.route('/api/files/<path:filename>', methods=['DELETE'])
    def delete_file(filename):
        """Delete a specific file"""
        try:
            # Validate filename
            is_valid, error = PathValidator.validate_filename(filename)
            if not is_valid:
                return jsonify({"error": error}), 400

            # Delete file
            deleted = file_service.delete_file(filename)

            if deleted:
                current_app.logger.info(f"File deleted: {filename}")
                return jsonify({"success": True, "message": f"File {filename} deleted successfully"})
            else:
                return jsonify({"error": "File not found"}), 404

        except Exception as e:
            current_app.logger.error(f"Error deleting file {filename}: {str(e)}")
            return jsonify({"error": "Delete failed", "details": str(e)}), 500

    @bp.route('/api/files/batch/download', methods=['POST'])
    def batch_download_files():
        """Download multiple files as a zip archive"""
        try:
            # Get and validate filenames
            data = request.json
            if not data or 'filenames' not in data:
                return jsonify({"error": "No filenames provided"}), 400

            filenames = data['filenames']
            is_valid, error = PathValidator.validate_filenames(filenames)
            if not is_valid:
                return jsonify({"error": error}), 400

            # Create in-memory zip file
            zip_buffer = io.BytesIO()

            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                files_added = 0

                for filename in filenames:
                    file_path = file_service.find_file(filename)
                    if file_path:
                        zip_file.write(file_path, filename)
                        files_added += 1

            if files_added == 0:
                return jsonify({"error": "No valid files found to download"}), 404

            # Prepare zip for download
            zip_buffer.seek(0)

            # Generate filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            zip_filename = f"translated_files_{timestamp}.zip"

            return send_file(
                zip_buffer,
                mimetype='application/zip',
                as_attachment=True,
                download_name=zip_filename
            )

        except Exception as e:
            current_app.logger.error(f"Error creating batch download: {str(e)}")
            return jsonify({"error": "Batch download failed", "details": str(e)}), 500

    @bp.route('/api/files/batch/delete', methods=['POST'])
    def batch_delete_files():
        """Delete multiple files"""
        try:
            # Get and validate filenames
            data = request.json
            if not data or 'filenames' not in data:
                return jsonify({"error": "No filenames provided"}), 400

            filenames = data['filenames']
            is_valid, error = PathValidator.validate_filenames(filenames)
            if not is_valid:
                return jsonify({"error": error}), 400

            deleted_files = []
            failed_files = []

            for filename in filenames:
                try:
                    deleted = file_service.delete_file(filename)
                    if deleted:
                        deleted_files.append(filename)
                    else:
                        failed_files.append({"filename": filename, "reason": "File not found"})
                except Exception as e:
                    failed_files.append({"filename": filename, "reason": str(e)})

            return jsonify({
                "success": True,
                "deleted": deleted_files,
                "failed": failed_files,
                "total_deleted": len(deleted_files)
            })

        except Exception as e:
            current_app.logger.error(f"Error in batch delete: {str(e)}")
            return jsonify({"error": "Batch delete failed", "details": str(e)}), 500

    @bp.route('/api/uploads/clear', methods=['POST'])
    def clear_uploaded_files():
        """Delete uploaded files based on their paths"""
        try:
            # Get and validate file paths
            data = request.json
            if not data or 'file_paths' not in data:
                return jsonify({"error": "No file paths provided"}), 400

            file_paths = data['file_paths']
            if not isinstance(file_paths, list):
                return jsonify({"error": "Invalid file paths list"}), 400

            deleted_files = []
            failed_files = []

            for file_path_str in file_paths:
                success, error = file_service.delete_uploaded_file(file_path_str)
                if success:
                    deleted_files.append(file_path_str)
                    current_app.logger.info(f"Deleted uploaded file: {file_path_str}")
                else:
                    failed_files.append({"file_path": file_path_str, "reason": error})

            return jsonify({
                "success": True,
                "deleted": deleted_files,
                "failed": failed_files,
                "total_deleted": len(deleted_files)
            })

        except Exception as e:
            current_app.logger.error(f"Error clearing uploaded files: {str(e)}")
            return jsonify({"error": "Clear uploads failed", "details": str(e)}), 500

    @bp.route('/api/files/<path:filename>/open', methods=['POST'])
    def open_local_file(filename):
        """Open a file in the default system application"""
        try:
            # Validate filename
            is_valid, error = PathValidator.validate_filename(filename)
            if not is_valid:
                return jsonify({"error": error}), 400

            # Open file
            success, message, abs_path = file_service.open_file(filename)

            if success:
                current_app.logger.info(f"Opened file: {filename} at {abs_path}")
                return jsonify({
                    "success": True,
                    "message": message,
                    "file_path": abs_path
                })
            else:
                current_app.logger.error(f"Error opening file {filename}: {message}")
                if "not allowed" in message.lower():
                    return jsonify({"error": message}), 403
                return jsonify({"error": message}), 404 if "not found" in message.lower() else 500

        except Exception as e:
            current_app.logger.error(f"Error in open_local_file for {filename}: {str(e)}")
            return jsonify({"error": "Failed to open file", "details": str(e)}), 500

    @bp.route('/api/folders/output/open', methods=['POST'])
    def open_output_folder():
        """Open the translations output folder in the system's file explorer."""
        try:
            success, message, abs_path = file_service.open_output_folder()
            if success:
                current_app.logger.info(f"Opened output folder: {abs_path}")
                return jsonify({"success": True, "message": message, "folder_path": abs_path})
            current_app.logger.error(f"Error opening output folder: {message}")
            return jsonify({"error": message, "folder_path": abs_path}), 500
        except Exception as e:
            current_app.logger.error(f"Error in open_output_folder: {str(e)}")
            return jsonify({"error": "Failed to open output folder", "details": str(e)}), 500

    @bp.route('/api/files/<path:filename>/reveal', methods=['POST'])
    def reveal_local_file(filename):
        """Reveal a file in the system's file explorer (selecting it when possible)"""
        try:
            is_valid, error = PathValidator.validate_filename(filename)
            if not is_valid:
                return jsonify({"error": error}), 400

            success, message, abs_path = file_service.reveal_file(filename)

            if success:
                current_app.logger.info(f"Revealed file: {filename} at {abs_path}")
                return jsonify({
                    "success": True,
                    "message": message,
                    "file_path": abs_path
                })
            else:
                current_app.logger.error(f"Error revealing file {filename}: {message}")
                return jsonify({"error": message}), 404 if "not found" in message.lower() else 500

        except Exception as e:
            current_app.logger.error(f"Error in reveal_local_file for {filename}: {str(e)}")
            return jsonify({"error": "Failed to reveal file", "details": str(e)}), 500

    return bp
