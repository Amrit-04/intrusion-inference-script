#!/usr/bin/env python3
"""
Simple script to test MJPEG stream connection from Raspberry Pi
"""
import cv2
import time

# Your Pi's MJPEG stream URL
MJPEG_URL = "http://10.120.180.108:8080/?action=stream&dummy=1"

def test_stream():
    print(f"Testing connection to: {MJPEG_URL}")
    
    # Try to connect
    cap = cv2.VideoCapture(MJPEG_URL)
    
    if not cap.isOpened():
        print("❌ Failed to connect to stream")
        print("Please check:")
        print("1. Raspberry Pi is running and accessible")
        print("2. MJPEG streamer is running on the Pi")
        print("3. IP address and port are correct")
        return False
    
    print("✅ Successfully connected to stream")
    
    # Test reading frames
    frame_count = 0
    start_time = time.time()
    
    print("Testing frame capture... (Press 'q' to quit)")
    
    while True:
        ret, frame = cap.read()
        
        if not ret or frame is None:
            print("❌ Failed to read frame")
            break
            
        frame_count += 1
        
        # Calculate FPS
        elapsed = time.time() - start_time
        if elapsed > 0:
            fps = frame_count / elapsed
        else:
            fps = 0
            
        # Add info to frame
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Resolution: {frame.shape[1]}x{frame.shape[0]}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Stream Test", frame)
        
        # Print stats every 30 frames
        if frame_count % 30 == 0:
            print(f"Frames received: {frame_count}, FPS: {fps:.1f}, Resolution: {frame.shape[1]}x{frame.shape[0]}")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\nTest completed. Total frames: {frame_count}")
    return True

if __name__ == "__main__":
    test_stream()