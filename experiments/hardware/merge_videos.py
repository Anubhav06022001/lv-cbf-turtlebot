# from moviepy import VideoFileClip, CompositeVideoClip

# # 1. Load both videos
# mobile_clip = VideoFileClip("/home/anubhav/Documents/lv-cbf-turtle_bot/experiments/hardware/lvcbf2.mp4")
# realsense_clip = VideoFileClip("/home/anubhav/Documents/lv-cbf-turtle_bot/experiments/hardware/autonomous_demo.mp4")


# realsense_pip = (realsense_clip
#                  .resized(width=480)
#                  .with_position(("right", "top")))

# # 4. Layer them together
# final_video = CompositeVideoClip([mobile_clip, realsense_pip])

# # 5. Export the final file
# print("Merging videos... this might take a few minutes!")
# final_video.write_videofile("/home/anubhav/Documents/lv-cbf-turtle_bot/experiments/hardware/presentation_video.mp4", codec="libx264")




from moviepy import VideoFileClip, CompositeVideoClip

# 1. Load both videos
mobile_clip = VideoFileClip("/home/anubhav/Documents/lv-cbf-turtle_bot/experiments/hardware/lvcbf2.mp4")
realsense_clip = VideoFileClip("/home/anubhav/Documents/lv-cbf-turtle_bot/experiments/hardware/autonomous_demo.mp4")

# Optional: Add .subclipped(start_time=X) here if you need to synchronize them

# 2. Force the mobile video to standard 1080p height and center it
mobile_clip = mobile_clip.resized(height=1080).with_position("center")

# 3. Create the Picture-in-Picture effect
# Since we have lots of empty space on the right, we can make the RealSense video much bigger!
realsense_pip = (realsense_clip
                 .resized(width=540) 
                 .with_position(("right", "top"))) # Places it in the black space on the right side

# 4. Layer them together on a forced 1920x1080 horizontal canvas
final_video = CompositeVideoClip([mobile_clip, realsense_pip], size=(1920, 1080))

# Ensure the video ends exactly when the mobile recording ends
final_video = final_video.with_duration(mobile_clip.duration)

# 5. Export the final file
print("Merging videos into a horizontal presentation format...")
final_video.write_videofile("/home/anubhav/Documents/lv-cbf-turtle_bot/experiments/hardware/presentation_video3.mp4", codec="libx264")