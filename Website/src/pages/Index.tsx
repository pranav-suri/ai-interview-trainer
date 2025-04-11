
import React from 'react';
import { Button } from '@/components/ui/button';
import { Link } from 'react-router-dom';
import Header from '@/components/layout/Header';
import Footer from '@/components/layout/Footer';
import { 
  Video, 
  BarChart3, 
  CheckCircle, 
  Brain, 
  ChevronRight,
  Mic,
  LineChart,
  PencilRuler
} from 'lucide-react';

const Index = () => {
  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      
      <main className="flex-grow">
        {/* Hero Section */}
        <section className="bg-interview-background py-20">
          <div className="container mx-auto px-4">
            <div className="flex flex-col md:flex-row items-center gap-12">
              <div className="md:w-1/2 space-y-6">
                <h1 className="text-4xl md:text-5xl font-bold leading-tight">
                  Ace Your Job Interviews with <span className="gradient-text">AI-Powered Coaching</span>
                </h1>
                <p className="text-lg text-gray-700">
                  Practice interviews with real-time feedback on your facial expressions, tone of voice, and speech patterns to build your confidence and get hired faster.
                </p>
                <div className="flex gap-4 pt-4">
                  <Button asChild className="bg-interview-primary hover:bg-interview-primary/90 text-white px-8">
                    <Link to="/practice">Start Practicing</Link>
                  </Button>
                  <Button variant="outline" asChild>
                    <Link to="/about">Learn More</Link>
                  </Button>
                </div>
              </div>
              <div className="md:w-1/2">
                <img 
                  src="/placeholder.svg" 
                  alt="AI Interview Coach" 
                  className="rounded-lg shadow-xl w-full max-w-lg mx-auto" 
                />
              </div>
            </div>
          </div>
        </section>
        
        {/* Features Section */}
        <section className="py-20">
          <div className="container mx-auto px-4">
            <h2 className="text-3xl font-bold text-center mb-12">How InterviewCoach Helps You</h2>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
              <div className="feature-card">
                <Video className="w-12 h-12 text-interview-primary mb-4" />
                <h3 className="text-xl font-semibold mb-2">Video Analysis</h3>
                <p className="text-gray-600">
                  Our AI analyzes your facial expressions and body language to measure confidence and engagement.
                </p>
              </div>
              
              <div className="feature-card">
                <Mic className="w-12 h-12 text-interview-secondary mb-4" />
                <h3 className="text-xl font-semibold mb-2">Voice Assessment</h3>
                <p className="text-gray-600">
                  Get feedback on filler words, speech clarity, and speaking pace to improve your communication.
                </p>
              </div>
              
              <div className="feature-card">
                <BarChart3 className="w-12 h-12 text-interview-accent mb-4" />
                <h3 className="text-xl font-semibold mb-2">Detailed Scoring</h3>
                <p className="text-gray-600">
                  Receive comprehensive performance metrics and targeted improvement suggestions.
                </p>
              </div>
            </div>
          </div>
        </section>
        
        {/* How It Works Section */}
        <section className="bg-gray-50 py-20">
          <div className="container mx-auto px-4">
            <h2 className="text-3xl font-bold text-center mb-12">How It Works</h2>
            
            <div className="space-y-12 max-w-3xl mx-auto">
              <div className="flex items-start gap-6">
                <div className="bg-interview-primary rounded-full p-3 text-white shrink-0">
                  <PencilRuler className="w-6 h-6" />
                </div>
                <div>
                  <h3 className="text-xl font-semibold mb-2">Choose Your Interview Type</h3>
                  <p className="text-gray-600">
                    Select from various interview types like behavioral, technical, or industry-specific questions.
                  </p>
                </div>
              </div>
              
              <div className="flex items-start gap-6">
                <div className="bg-interview-primary rounded-full p-3 text-white shrink-0">
                  <Video className="w-6 h-6" />
                </div>
                <div>
                  <h3 className="text-xl font-semibold mb-2">Record Your Practice Session</h3>
                  <p className="text-gray-600">
                    Answer interview questions in real-time while our AI tracks your facial expressions and voice.
                  </p>
                </div>
              </div>
              
              <div className="flex items-start gap-6">
                <div className="bg-interview-primary rounded-full p-3 text-white shrink-0">
                  <Brain className="w-6 h-6" />
                </div>
                <div>
                  <h3 className="text-xl font-semibold mb-2">Get AI-Powered Feedback</h3>
                  <p className="text-gray-600">
                    Receive insights on your performance, including confidence ratings, engagement levels, and speech patterns.
                  </p>
                </div>
              </div>
              
              <div className="flex items-start gap-6">
                <div className="bg-interview-primary rounded-full p-3 text-white shrink-0">
                  <LineChart className="w-6 h-6" />
                </div>
                <div>
                  <h3 className="text-xl font-semibold mb-2">Track Your Improvement</h3>
                  <p className="text-gray-600">
                    See your progress over time and focus on specific areas that need improvement.
                  </p>
                </div>
              </div>
            </div>
            
            <div className="text-center mt-12">
              <Button asChild className="bg-interview-primary hover:bg-interview-primary/90 text-white px-8">
                <Link to="/practice">
                  Try It Now <ChevronRight className="ml-2 h-4 w-4" />
                </Link>
              </Button>
            </div>
          </div>
        </section>
        
        {/* Testimonials Section */}
        <section className="py-20">
          <div className="container mx-auto px-4">
            <h2 className="text-3xl font-bold text-center mb-12">Success Stories</h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
              <div className="bg-white p-6 rounded-xl shadow-md">
                <div className="flex items-center mb-4">
                  <div className="w-12 h-12 rounded-full bg-gray-200 mr-4"></div>
                  <div>
                    <h4 className="font-semibold">Michael L.</h4>
                    <p className="text-sm text-gray-500">Software Engineer</p>
                  </div>
                </div>
                <p className="text-gray-600">
                  "After just two weeks of practice, I felt so much more confident in my technical interviews. The feedback on my nervous habits was eye-opening!"
                </p>
              </div>
              
              <div className="bg-white p-6 rounded-xl shadow-md">
                <div className="flex items-center mb-4">
                  <div className="w-12 h-12 rounded-full bg-gray-200 mr-4"></div>
                  <div>
                    <h4 className="font-semibold">Sarah J.</h4>
                    <p className="text-sm text-gray-500">Marketing Director</p>
                  </div>
                </div>
                <p className="text-gray-600">
                  "The AI feedback helped me realize I was using too many filler words. After practicing, I landed my dream job at a top agency!"
                </p>
              </div>
              
              <div className="bg-white p-6 rounded-xl shadow-md">
                <div className="flex items-center mb-4">
                  <div className="w-12 h-12 rounded-full bg-gray-200 mr-4"></div>
                  <div>
                    <h4 className="font-semibold">David T.</h4>
                    <p className="text-sm text-gray-500">Financial Analyst</p>
                  </div>
                </div>
                <p className="text-gray-600">
                  "I was always nervous during interviews. This tool helped me track my improvement and build confidence over time."
                </p>
              </div>
            </div>
          </div>
        </section>
        
        {/* CTA Section */}
        <section className="bg-interview-primary text-white py-16">
          <div className="container mx-auto px-4 text-center">
            <h2 className="text-3xl font-bold mb-6">Ready to Ace Your Next Interview?</h2>
            <p className="text-xl mb-8 max-w-2xl mx-auto">
              Start practicing today and get personalized feedback to help you land your dream job.
            </p>
            <Button asChild size="lg" variant="secondary" className="bg-white text-interview-primary hover:bg-gray-100">
              <Link to="/practice">Get Started For Free</Link>
            </Button>
          </div>
        </section>
      </main>
      
      <Footer />
    </div>
  );
};

export default Index;
