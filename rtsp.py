import cv2
import os
import time
import argparse
import sys

def test_rtsp_connection(rtsp_url):
    """Test RTSP connection using multiple methods"""
    methods = [
        ("Standard OpenCV", lambda: cv2.VideoCapture(rtsp_url)),
        ("FFMPEG with TCP", lambda: setup_ffmpeg_tcp_and_capture(rtsp_url)),
        ("FFMPEG with UDP", lambda: setup_ffmpeg_udp_and_capture(rtsp_url)),
        ("FFMPEG with extended options", lambda: setup_ffmpeg_extended_and_capture(rtsp_url))
    ]
    
    # Add GStreamer method only if supported
    if 'GStreamer' in cv2.getBuildInformation():
        methods.append(("GStreamer pipeline", lambda: setup_gstreamer_and_capture(rtsp_url)))
    
    print(f"\nTesting RTSP connection to {rtsp_url}")
    for name, method in methods:
        print(f"\n----- Trying {name} -----")
        success = try_connection_method(method)
        if success:
            print(f"SUCCESS: {name} worked!")
            return True
        else:
            print(f"FAILED: {name} didn't work.")
    
    print("\nAll connection methods failed.")
    return False

def setup_ffmpeg_tcp_and_capture(rtsp_url):
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    return cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

def setup_ffmpeg_udp_and_capture(rtsp_url):
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
    return cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

def setup_ffmpeg_extended_and_capture(rtsp_url):
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|buffer_size;1024000|stimeout;5000000|max_delay;500000"
    return cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

def setup_gstreamer_and_capture(rtsp_url):
    gst_str = (f'rtspsrc location={rtsp_url} latency=0 ! '
               'rtph264depay ! h264parse ! avdec_h264 ! '
               'videoconvert ! appsink')
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

def try_connection_method(method):
    try:
        cap = method()
        if not cap.isOpened():
            print("  Connection failed: Camera not opened")
            return False
            
        # Try to read frames
        frame_count = 0
        start_time = time.time()
        
        while time.time() - start_time < 5:  # Try for 5 seconds
            ret, frame = cap.read()
            if not ret:
                print("  Failed to read frame")
                time.sleep(0.1)  # Small delay between attempts
                continue
                
            if frame is None or frame.size == 0:
                print("  Received empty frame")
                time.sleep(0.1)
                continue
                
            frame_count += 1
            if frame_count == 1:
                print(f"  First frame received! Size: {frame.shape}")
                
            if frame_count % 10 == 0:
                print(f"  Read {frame_count} frames...")
        
        cap.release()
        
        if frame_count > 0:
            print(f"  Successfully read {frame_count} frames in {time.time() - start_time:.2f} seconds")
            print(f"  Frame rate: ~{frame_count / (time.time() - start_time):.1f} FPS")
            return True
        else:
            print("  No frames could be read")
            return False
            
    except Exception as e:
        print(f"  Exception: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test RTSP connection with multiple methods")
    parser.add_argument("--url", default="rtsp://test123:test123@192.168.1.23:554/stream1", help="RTSP URL")
    args = parser.parse_args()
    
    print(f"OpenCV version: {cv2.__version__}")
    print(f"GStreamer in build info: {'Yes' if 'GStreamer' in cv2.getBuildInformation() else 'No'}")
    
    success = test_rtsp_connection(args.url)
    sys.exit(0 if success else 1)