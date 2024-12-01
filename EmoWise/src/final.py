import pickle
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model("emotion_model.h5")

# Define emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Save model and labels into a Pickle file
data_to_save = {
    "model": model,
    "emotion_labels": emotion_labels,
}

with open("emotiondata_load.pkl", "wb") as file:
    pickle.dump(data_to_save, file)

print("Pickle file created successfully!")



import streamlit as st
import cv2
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Load your model
try:
    with open("emotiondata_load.pkl", "rb") as file:
        loaded_data = pickle.load(file)
        model = loaded_data["model"]
        emotion_labels = loaded_data["emotion_labels"]
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# Define recommendations
# Define expanded recommendations
recommendations = {
    'angry': (
        "Take a deep breath and relax. Consider listening to calming music or practicing mindfulness. "
        "You might also try these activities: "
        "- Go for a brisk walk or jog to release energy. "
        "- Practice yoga or meditation to center yourself. "
        "- Try a creative outlet like painting, writing, or playing a musical instrument."
    ),
    'disgust': (
        "Try to focus on something you enjoy or take a walk to clear your mind. "
        "Here are additional activities to consider: "
        "- Watch a feel-good movie or comedy to change your mood. "
        "- Engage in gardening or taking care of plants. "
        "- Learn a new skill like cooking or crafting."
    ),
    'fear': (
        "It’s okay to feel scared. Talk to someone you trust or write your thoughts in a journal. "
        "Other ways to cope include: "
        "- Reading a book or listening to an inspiring podcast. "
        "- Doing a light physical activity like stretching or yoga. "
        "- Practicing breathing exercises to calm your nerves."
    ),
    'happy': (
        "Keep smiling! Share your happiness with others or capture the moment in a photo. "
        "You can also: "
        "- Call a friend or family member to share your joy. "
        "- Dance to your favorite songs or create a playlist of happy music. "
        "- Try journaling about what made you happy today."
    ),
    'neutral': (
        "A neutral state is great for focusing. Take this opportunity to plan or organize your day. "
        "Other activities to consider: "
        "- Start a new project or hobby you’ve been meaning to try. "
        "- Organize your living space or workspace for productivity. "
        "- Listen to an audiobook or learn something new online."
    ),
    'sad': (
        "It’s okay to feel sad. Listen to your favorite music or call a loved one for support. "
        "You could also try: "
        "- Watching a comforting movie or TV series. "
        "- Writing your thoughts in a journal to process your emotions. "
        "- Doing a small act of kindness for someone else to uplift your spirit."
    ),
    'surprise': (
        "Enjoy the moment! Share your excitement or take a break to process the surprise. "
        "Consider these activities as well: "
        "- Capture the moment with a photo or video. "
        "- Share the surprise with a close friend or family member. "
        "- Channel the energy into a spontaneous activity, like trying a new recipe or going for an adventure."
    )
}


# Streamlit Title
st.title("Real-Time Emotion Detection with Personalized Recommendations")
st.write("This application uses your webcam to detect emotions and provide recommendations.")

# Webcam button
if st.button("Start Webcam Emotion Detection"):
    st.warning("Press 'q' to stop the webcam.")

    # Initialize webcam
    cap = cv2.VideoCapture(0)  # 0 is the default camera

    # Load Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access the webcam. Please check your camera.")
            break

        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Extract the face region
            roi_gray = gray_frame[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48)) / 255.0  # Normalize
            roi_gray = np.expand_dims(roi_gray, axis=-1)  # Add channel dimension
            roi_gray = np.expand_dims(roi_gray, axis=0)  # Add batch dimension

            # Predict emotion
            predictions = model.predict(roi_gray)
            emotion_index = np.argmax(predictions)
            detected_emotion = emotion_labels[emotion_index]

            # Draw rectangle and label on the frame
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, detected_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # Display recommendation
            st.write(f"**Detected Emotion:** {detected_emotion.capitalize()}")
            st.write(f"**Personalized Recommendation:** {recommendations[detected_emotion]}")

        # Display the frame
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        # Stop when 'q' is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
