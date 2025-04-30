import cv2
import numpy as np

def check_opencv_info():
    print(f"OpenCV version: {cv2.__version__}")
    
    # Check if OpenCV was built with GStreamer
    gstreamer_support = 'GStreamer' if cv2.getBuildInformation().find('GStreamer') != -1 else 'No GStreamer'
    print(f"GStreamer support: {gstreamer_support}")
    
    # List available backends
    backends = [cv2.CAP_ANY, cv2.CAP_VFW, cv2.CAP_V4L, cv2.CAP_V4L2, 
                cv2.CAP_FIREWIRE, cv2.CAP_FIREWARE, cv2.CAP_IEEE1394, 
                cv2.CAP_DC1394, cv2.CAP_CMU1394, cv2.CAP_DSHOW, 
                cv2.CAP_MSMF, cv2.CAP_GSTREAMER]
                
    print("\nAvailable backends:")
    for backend in backends:
        try:
            cap = cv2.VideoCapture(0, backend)
            is_available = cap.isOpened()
            cap.release()
            backend_name = {
                cv2.CAP_ANY: "Auto",
                cv2.CAP_VFW: "VFW",
                cv2.CAP_V4L: "V4L",
                cv2.CAP_V4L2: "V4L2",
                cv2.CAP_FIREWIRE: "FireWire",
                cv2.CAP_FIREWARE: "FireWare",
                cv2.CAP_IEEE1394: "IEEE 1394",
                cv2.CAP_DC1394: "DC1394",
                cv2.CAP_CMU1394: "CMU1394",
                cv2.CAP_DSHOW: "DirectShow",
                cv2.CAP_MSMF: "Microsoft Media Foundation",
                cv2.CAP_GSTREAMER: "GStreamer"
            }.get(backend, f"Unknown ({backend})")
            
            print(f"- {backend_name}: {'Available' if is_available else 'Not available'}")
        except:
            print(f"- Backend {backend}: Error checking availability")
            
if __name__ == "__main__":
    check_opencv_info()