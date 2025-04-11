
export interface InterviewQuestion {
  id: number;
  question: string;
  category: 'Behavioral' | 'Technical' | 'Experience' | 'Situational' | 'Personal';
  difficulty: 'Easy' | 'Medium' | 'Hard';
}

export const interviewQuestions: InterviewQuestion[] = [
  {
    id: 1,
    question: "Tell me about yourself and your background.",
    category: "Personal",
    difficulty: "Easy"
  },
  {
    id: 2,
    question: "Why do you want to work for this company?",
    category: "Behavioral",
    difficulty: "Easy"
  },
  {
    id: 3,
    question: "Describe a time when you had to overcome a significant challenge at work.",
    category: "Behavioral",
    difficulty: "Medium"
  },
  {
    id: 4,
    question: "What are your greatest strengths and weaknesses?",
    category: "Personal",
    difficulty: "Medium"
  },
  {
    id: 5,
    question: "Tell me about a time you had to work with a difficult team member.",
    category: "Situational",
    difficulty: "Medium"
  },
  {
    id: 6,
    question: "Where do you see yourself in five years?",
    category: "Personal",
    difficulty: "Easy"
  },
  {
    id: 7,
    question: "Describe a project you're particularly proud of and your role in it.",
    category: "Experience",
    difficulty: "Medium"
  },
  {
    id: 8,
    question: "How do you handle pressure or stressful situations?",
    category: "Behavioral",
    difficulty: "Medium"
  },
  {
    id: 9,
    question: "Tell me about a time you failed and what you learned from it.",
    category: "Behavioral",
    difficulty: "Hard"
  },
  {
    id: 10,
    question: "How do you prioritize your work when dealing with multiple deadlines?",
    category: "Situational",
    difficulty: "Medium"
  },
  {
    id: 11,
    question: "Why are you leaving your current position?",
    category: "Personal",
    difficulty: "Medium"
  },
  {
    id: 12,
    question: "Describe your ideal work environment.",
    category: "Personal",
    difficulty: "Easy"
  },
  {
    id: 13,
    question: "How do you stay updated with industry trends and developments?",
    category: "Technical",
    difficulty: "Medium"
  },
  {
    id: 14,
    question: "Tell me about a time you had to make a difficult decision with limited information.",
    category: "Situational",
    difficulty: "Hard"
  },
  {
    id: 15,
    question: "What motivates you to do your best work?",
    category: "Personal",
    difficulty: "Medium"
  }
];

// Helper function to get a random question
export const getRandomQuestion = (): InterviewQuestion => {
  const randomIndex = Math.floor(Math.random() * interviewQuestions.length);
  return interviewQuestions[randomIndex];
};

// Helper function to get a random question from a specific category
export const getRandomQuestionByCategory = (category: InterviewQuestion['category']): InterviewQuestion => {
  const filteredQuestions = interviewQuestions.filter(q => q.category === category);
  
  if (filteredQuestions.length === 0) {
    return getRandomQuestion();
  }
  
  const randomIndex = Math.floor(Math.random() * filteredQuestions.length);
  return filteredQuestions[randomIndex];
};

// Helper function to get a curated set of questions for a mock interview
export const getInterviewSet = (questionCount: number = 5): InterviewQuestion[] => {
  // Ensure we don't ask for more questions than are available
  const count = Math.min(questionCount, interviewQuestions.length);
  
  // Get a random subset of questions
  const shuffled = [...interviewQuestions].sort(() => 0.5 - Math.random());
  return shuffled.slice(0, count);
};
