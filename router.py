from database import app
from app import upload_rtsp_data, update_rtsp_data, get_rtsp_data, delete_rtsp_data, process_rtsp_data, list_files, serve_file, delete_all_files, api_test
from liveFeeds import video_feed, index

# Register routes
app.add_url_rule('/upload_rtsp_data', view_func=upload_rtsp_data, methods=['POST'])
app.add_url_rule('/update_rtsp_data/<int:id>', view_func=update_rtsp_data, methods=['PUT'])
app.add_url_rule('/get_rtsp_data', view_func=get_rtsp_data, methods=['GET'])
app.add_url_rule('/get_rtsp_data/<int:id>', view_func=get_rtsp_data, methods=['GET'])
app.add_url_rule('/delete_rtsp_data/<int:id>', view_func=delete_rtsp_data, methods=['DELETE'])
app.add_url_rule("/process_rtsp_data/<int:id>", view_func=process_rtsp_data, methods=['POST'])
app.add_url_rule("/files/<folder>", view_func=list_files, methods=['GET'])
app.add_url_rule("/files/<folder>/<filename>", view_func=serve_file, methods=['GET'])
app.add_url_rule("/delete_all_files", view_func=delete_all_files, methods=['DELETE'])
app.add_url_rule("/test-api", view_func=api_test, methods=['POST'])
app.add_url_rule("/video_feed/<int:id>", view_func=video_feed, methods=['GET'])
app.add_url_rule("/stream/<int:id>", view_func=index, methods=['GET'])



if __name__ == "__main__":
    with app.app_context():
        from database import db
        db.create_all()
        # Run the app on port 8000
        app.run(debug=True, host='0.0.0.0', port=8000)