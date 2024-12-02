import streamlit as st

import pandas as pd
import numpy as np
import altair as alt

import joblib

pipe_lr = joblib.load(open("models/text_emotion.pkl", "rb"))

emotions_emoji_dict = {"anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", "joy": "😂", "neutral": "😐", "sad": "😔",
                       "sadness": "😔", "shame": "😳", "surprise": "😮"}

emotions_recommendation_dict = {
    "anger": {
        "hobbies": ["Meditation", "Yoga", "Journaling", "Painting"],
        "recipes": ["Chamomile tea", "Soothing soup", "Dark chocolate treat"]
    },
    "disgust": {
        "hobbies": ["Gardening", "Walking in nature", "Watching a comedy"],
        "recipes": ["Fresh fruit salad", "Lemonade", "Herbal detox drink"]
    },
    "fear": {
        "hobbies": ["Reading a comforting book", "Stretching", "Deep breathing exercises"],
        "recipes": ["Warm milk", "Honey ginger tea", "Comfort food like mac and cheese"]
    },
    "happy": {
        "hobbies": ["Dancing", "Cooking something new", "Calling a friend"],
        "recipes": ["Celebratory cupcakes", "Fruit smoothie", "Homemade pizza"]
    },
    "joy": {
        "hobbies": ["Playing a game", "Sharing a story", "Baking"],
        "recipes": ["Cheesecake", "Mango salsa", "Berry parfait"]
    },
    "neutral": {
        "hobbies": ["Planning a day trip", "Organizing your space", "Listening to music"],
        "recipes": ["Avocado toast", "Simple salad", "Green tea"]
    },
    "sad": {
        "hobbies": ["Writing poetry", "Listening to uplifting music", "Walking outdoors"],
        "recipes": ["Chocolate chip cookies", "Hot cocoa", "Comforting soup"]
    },
    "shame": {
        "hobbies": ["Reflecting in a journal", "Doing light exercise", "Calling a trusted friend"],
        "recipes": ["Oatmeal with berries", "Chamomile tea", "Banana bread"]
    },
    "surprise": {
        "hobbies": ["Exploring a new hobby", "Trying a new recipe", "Photography"],
        "recipes": ["Exotic stir-fry", "Ice cream sundae", "Homemade pasta"]
    }
}

def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]


def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results


def main():
    st.title("Hey Buddy!!I am here for you")
    st.subheader("What's your current state of mind?")

    with st.form(key='my_form'):
        raw_text = st.text_area("Share Here")
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
            #st.write("Confidence:{}".format(np.max(probability)))

        with col2:
            #st. success("Prediction Probability")
            #st.write(probability)
            #proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
            #st.write(proba_df.T)
            #proba_df_clean = proba_df.T.reset_index()
            #proba_df_clean.columns = ["emotions", "probability"]

            #fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
            #st.altair_chart(fig, use_container_width=True)

            #st.subheader("Personalized Recommendation")
            #recommendation = emotions_recommendation_dict.get(prediction, "Take care and try to understand your feelings.")
            #st.info(recommendation)

            #st.subheader("Personalized Recommendations")
            recommendations = emotions_recommendation_dict.get(prediction, {
            "hobbies": ["Take some time for self-care"],
            "recipes": ["Enjoy a favorite meal or snack"]
        })

        st.info(f"### Hobbies to Try:")
        for hobby in recommendations["hobbies"]:
            st.write(f"- {hobby}")

        st.info(f"### Recipes to Explore:")
        for recipe in recommendations["recipes"]:
            st.write(f"- {recipe}")


if __name__ == '__main__':
    main()


