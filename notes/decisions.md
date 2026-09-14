# Decisions
- Start date: 2026-09-14
- Hours/day target: 4.5–5
- Pilot vertical: Maestro
- Fake first customer: Me
- Hardware: CPU

Product working name: Maestro
Buyer: Schools, Tuitions, Professional Development Teachers and Students
Job to be done: Given a variety of subjects, and some source material (academic papers, excerpts from books, previous question papers), generate realistic study plans and quizzes which gives teachers, students and other stakeholders (parents) insights into individual strengths and weaknesses.
What “good” looks like on Friday of week 5: Teacher uploads 3–10 source files. System proposes 10 questions, each with a citation into a chunk.You have 30 gold questions with expected source chunks. Metric: % of generated questions whose cited chunk actually supports the answer.
What the agent is allowed to do in week 8 (draft only vs write): The agent is able to continually generate weekly/daily quizzes that look at the source material and the mistakes of an individual. Imagine a personalised DuoLingo for whatever subject they're learning, where the source material and prompt is provided by the teacher and the teacher can prompt the agent to change behavior over time by monitoring student scores/outcomes.
What I will not build (billing, mobile app, multi-tenant SSO…): Authentication, Billing, Mobile apps, etc. This will be a very dirty product from an end-user point of view. My goal is to ensure the best improvements in students studying a subject via AI Engineering, any product engineering goodness will be excluded for the duration of this course.
First corpus: ICSE Grade 11-12 English Literature evaluation