
import React from 'react';
import { Card, CardContent } from '@/components/ui/card';
import Header from '@/components/layout/Header';
import Footer from '@/components/layout/Footer';
import { Button } from '@/components/ui/button';
import { CheckCircle, Users, Brain, Award, ChevronRight } from 'lucide-react';
import { Link } from 'react-router-dom';

const About = () => {
  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      
      <main className="flex-grow">
        {/* Hero Section */}
        <section className="bg-interview-background py-20">
          <div className="container mx-auto px-4 text-center">
            <h1 className="text-4xl md:text-5xl font-bold mb-6">About InterviewCoach</h1>
            <p className="text-xl text-gray-700 max-w-3xl mx-auto">
              We're on a mission to help job seekers build confidence and ace their interviews through AI-powered practice and feedback.
            </p>
          </div>
        </section>
        
        {/* Our Story Section */}
        <section className="py-20">
          <div className="container mx-auto px-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-12 items-center">
              <div>
                <h2 className="text-3xl font-bold mb-6">Our Story</h2>
                <p className="text-gray-700 mb-4">
                  InterviewCoach was born from a simple observation: most people struggle with interview anxiety, and traditional preparation methods don't provide real-time feedback.
                </p>
                <p className="text-gray-700 mb-4">
                  Our team of career coaches, AI researchers, and software developers came together to create a tool that could analyze the subtle aspects of interview performance that make the difference between getting hired and being passed over.
                </p>
                <p className="text-gray-700">
                  Today, we've helped thousands of job seekers improve their interview skills and land their dream jobs through targeted, personalized feedback and coaching.
                </p>
              </div>
              <div className="flex justify-center">
                <img 
                  src="/placeholder.svg" 
                  alt="Our Team" 
                  className="rounded-lg shadow-lg max-w-full h-auto" 
                />
              </div>
            </div>
          </div>
        </section>
        
        {/* How It Works Section */}
        <section className="bg-gray-50 py-20">
          <div className="container mx-auto px-4">
            <h2 className="text-3xl font-bold text-center mb-12">How Our Technology Works</h2>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
              <Card>
                <CardContent className="pt-6">
                  <div className="flex flex-col items-center text-center">
                    <div className="bg-interview-primary/10 text-interview-primary p-3 rounded-full mb-4">
                      <Brain className="w-10 h-10" />
                    </div>
                    <h3 className="text-xl font-semibold mb-3">AI Face Analysis</h3>
                    <p className="text-gray-600">
                      Our computer vision algorithms analyze micro-expressions and facial 
                      cues to determine confidence levels, engagement, and nervousness.
                    </p>
                  </div>
                </CardContent>
              </Card>
              
              <Card>
                <CardContent className="pt-6">
                  <div className="flex flex-col items-center text-center">
                    <div className="bg-interview-secondary/10 text-interview-secondary p-3 rounded-full mb-4">
                      <Users className="w-10 h-10" />
                    </div>
                    <h3 className="text-xl font-semibold mb-3">Voice Processing</h3>
                    <p className="text-gray-600">
                      Advanced audio analysis detects filler words, measures speaking pace, 
                      and evaluates tone to provide feedback on verbal communication.
                    </p>
                  </div>
                </CardContent>
              </Card>
              
              <Card>
                <CardContent className="pt-6">
                  <div className="flex flex-col items-center text-center">
                    <div className="bg-interview-accent/10 text-interview-accent p-3 rounded-full mb-4">
                      <Award className="w-10 h-10" />
                    </div>
                    <h3 className="text-xl font-semibold mb-3">Scoring System</h3>
                    <p className="text-gray-600">
                      Our proprietary scoring algorithm combines multiple factors to 
                      provide comprehensive feedback and actionable improvement suggestions.
                    </p>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        </section>
        
        {/* Benefits Section */}
        <section className="py-20">
          <div className="container mx-auto px-4">
            <h2 className="text-3xl font-bold text-center mb-12">Why Choose InterviewCoach</h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-x-12 gap-y-8">
              <div className="flex gap-4">
                <CheckCircle className="w-6 h-6 text-interview-primary shrink-0 mt-1" />
                <div>
                  <h3 className="text-xl font-semibold mb-2">Personalized Feedback</h3>
                  <p className="text-gray-600">
                    Get tailored insights specific to your interview performance and areas for improvement.
                  </p>
                </div>
              </div>
              
              <div className="flex gap-4">
                <CheckCircle className="w-6 h-6 text-interview-primary shrink-0 mt-1" />
                <div>
                  <h3 className="text-xl font-semibold mb-2">Practice Anytime</h3>
                  <p className="text-gray-600">
                    No scheduling or waiting for a coach - practice whenever you have time and as often as you need.
                  </p>
                </div>
              </div>
              
              <div className="flex gap-4">
                <CheckCircle className="w-6 h-6 text-interview-primary shrink-0 mt-1" />
                <div>
                  <h3 className="text-xl font-semibold mb-2">Track Your Progress</h3>
                  <p className="text-gray-600">
                    See how your interview skills improve over time with detailed metrics and charts.
                  </p>
                </div>
              </div>
              
              <div className="flex gap-4">
                <CheckCircle className="w-6 h-6 text-interview-primary shrink-0 mt-1" />
                <div>
                  <h3 className="text-xl font-semibold mb-2">Industry-Specific Questions</h3>
                  <p className="text-gray-600">
                    Practice with questions tailored to your industry and role for more relevant preparation.
                  </p>
                </div>
              </div>
              
              <div className="flex gap-4">
                <CheckCircle className="w-6 h-6 text-interview-primary shrink-0 mt-1" />
                <div>
                  <h3 className="text-xl font-semibold mb-2">AI-Powered Insights</h3>
                  <p className="text-gray-600">
                    Leverage cutting-edge technology that detects subtle cues human coaches might miss.
                  </p>
                </div>
              </div>
              
              <div className="flex gap-4">
                <CheckCircle className="w-6 h-6 text-interview-primary shrink-0 mt-1" />
                <div>
                  <h3 className="text-xl font-semibold mb-2">Privacy-Focused</h3>
                  <p className="text-gray-600">
                    Your practice sessions are processed locally and never stored permanently on our servers.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>
        
        {/* Team Section */}
        <section className="bg-gray-50 py-20">
          <div className="container mx-auto px-4">
            <h2 className="text-3xl font-bold text-center mb-12">Our Team</h2>
            
            <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
              <div className="text-center">
                <div className="w-40 h-40 rounded-full bg-gray-200 mx-auto mb-4"></div>
                <h3 className="text-xl font-semibold">Alex Chen</h3>
                <p className="text-gray-600">CEO & Co-Founder</p>
              </div>
              
              <div className="text-center">
                <div className="w-40 h-40 rounded-full bg-gray-200 mx-auto mb-4"></div>
                <h3 className="text-xl font-semibold">Sarah Johnson</h3>
                <p className="text-gray-600">Chief AI Officer</p>
              </div>
              
              <div className="text-center">
                <div className="w-40 h-40 rounded-full bg-gray-200 mx-auto mb-4"></div>
                <h3 className="text-xl font-semibold">Michael Lee</h3>
                <p className="text-gray-600">Lead Developer</p>
              </div>
              
              <div className="text-center">
                <div className="w-40 h-40 rounded-full bg-gray-200 mx-auto mb-4"></div>
                <h3 className="text-xl font-semibold">Jessica Rivera</h3>
                <p className="text-gray-600">Career Coach</p>
              </div>
            </div>
          </div>
        </section>
        
        {/* CTA Section */}
        <section className="bg-interview-primary text-white py-16">
          <div className="container mx-auto px-4 text-center">
            <h2 className="text-3xl font-bold mb-6">Ready to Transform Your Interview Skills?</h2>
            <p className="text-xl mb-8 max-w-2xl mx-auto">
              Join thousands of job seekers who have improved their interview performance with InterviewCoach.
            </p>
            <Button asChild size="lg" variant="secondary" className="bg-white text-interview-primary hover:bg-gray-100">
              <Link to="/practice">
                Start Practicing Now <ChevronRight className="ml-2 h-4 w-4" />
              </Link>
            </Button>
          </div>
        </section>
      </main>
      
      <Footer />
    </div>
  );
};

export default About;
