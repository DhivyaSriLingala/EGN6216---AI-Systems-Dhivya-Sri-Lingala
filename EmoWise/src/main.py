import streamlit as st

import pandas as pd
import numpy as np
import altair as alt

import joblib

pipe_lr = joblib.load(open("model/text_emotion.pkl", "rb"))

emotions_emoji_dict = {"anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", "joy": "😂", "neutral": "😐", "sad": "😔",
                       "sadness": "😔", "shame": "😳", "surprise": "😮"}

emotions_recommendation_dict = {
    "anger": "Take a few deep breaths and try a calming activity like yoga or meditation.",
    "disgust": "Distract yourself with a pleasant memory or enjoy a clean, refreshing drink.",
    "fear": "Try grounding techniques—focus on the details of your surroundings to feel safe.",
    "happy": "Celebrate your joy! Share it with someone close or write it in a journal.",
    "joy": "Spread the happiness! Consider doing something kind for someone else.",
    "neutral": "Take this calm moment to reflect or plan a small positive change.",
    "sad": "Reach out to a friend or listen to uplifting music to lighten your mood.",
    "sadness": "Remember, it's okay to feel this way. A warm drink or a favorite activity can help.",
    "shame": "Challenge negative thoughts—focus on your strengths and small victories.",
    "surprise": "Embrace the unexpected! Write down your thoughts or share your experience."
}



def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]


def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results


def main():
    st.title("Text Emotion Detection")
    st.subheader("Detect Emotions In Text")

    with st.form(key='my_form'):
        raw_text = st.text_area("Type Here")
        submit_text = st.form_submit_button(label='Submit')

    if submit_text:
        col1, col2 = st.columns(2)

        prediction = predict_emotions(raw_text)
        probability = get_prediction_proba(raw_text)

        with col1:
            st.success("Original Text")
            st.write(raw_text)

            st.success("Prediction")
            emoji_icon = emotions_emoji_dict[prediction]
            st.write("{}:{}".format(prediction, emoji_icon))
            st.write("Confidence:{}".format(np.max(probability)))

        with col2:
            st. success("Prediction Probability")
            #st.write(probability)
            proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
            #st.write(proba_df.T)
            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["emotions", "probability"]

            fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
            st.altair_chart(fig, use_container_width=True)

            st.subheader("Personalized Recommendation")
            recommendation = emotions_recommendation_dict.get(prediction, "Take care and try to understand your feelings.")
            st.info(recommendation)


if __name__ == '__main__':
    main()


