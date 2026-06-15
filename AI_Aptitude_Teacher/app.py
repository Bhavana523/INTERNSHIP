import streamlit as st
import pandas as pd
import os

# Website title
st.title("AI Aptitude Teacher")

# Website description
st.write("Practice aptitude questions with explanations")

# Read CSV file
questions = pd.read_csv("aptitude_questions.csv")

# Create score variable
score = 0

# Loop through each question
for index, row in questions.iterrows():

    # Display question number and question
    st.subheader(f"Question {row['id']}")
    st.write(row['question'])

    # Create options list
    options = [
        row['option_a'],
        row['option_b'],
        row['option_c'],
        row['option_d']
    ]

    # Radio button for options
    answer = st.radio(
        "Choose your answer",
        options,
        key=index
    )

    # Submit button
    if st.button(f"Submit Question {row['id']}"):

        # Find selected option letter
        if answer == row['option_a']:
            selected = "A"

        elif answer == row['option_b']:
            selected = "B"

        elif answer == row['option_c']:
            selected = "C"

        else:
            selected = "D"

        # Check answer
        if selected == row['correct_answer']:

            st.success("Correct Answer")

            result = "Correct"

            score += 1

        else:

            st.error("Wrong Answer")

            st.write(
                f"Correct Answer is {row['correct_answer']}"
            )

            result = "Wrong"

        # Show explanation
        st.info(f"Explanation: {row['explanation']}")

        # Show formula
        st.warning(f"Formula: {row['formula']}")

        # Create data dictionary
        progress = {
            "question": row['question'],
            "selected_answer": selected,
            "correct_answer": row['correct_answer'],
            "result": result,
            "topic": row['topic'],
            "difficulty": row['difficulty']
        }

        # Convert dictionary into dataframe
        progress_df = pd.DataFrame([progress])

        # Check if file exists
        file_exists = os.path.isfile(
            "student_progress.csv"
        )

        # Save into CSV
        progress_df.to_csv(
            "student_progress.csv",
            mode='a',
            header=not file_exists,
            index=False
        )

# Final score display
st.write("Quiz Completed")