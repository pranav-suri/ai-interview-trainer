
import React from 'react';
import { Link } from 'react-router-dom';

const Footer = () => {
  return (
    <footer className="bg-white border-t py-8">
      <div className="container mx-auto px-4">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          <div>
            <h3 className="font-bold text-lg mb-4 gradient-text">InterviewCoach</h3>
            <p className="text-sm text-gray-600">
              Helping job seekers ace their interviews with AI-powered feedback and coaching.
            </p>
          </div>
          <div>
            <h4 className="font-semibold mb-4">Features</h4>
            <ul className="space-y-2 text-sm">
              <li><Link to="/practice" className="text-gray-600 hover:text-interview-primary">Practice Interviews</Link></li>
              <li><Link to="/history" className="text-gray-600 hover:text-interview-primary">Progress Tracking</Link></li>
              <li><Link to="/about" className="text-gray-600 hover:text-interview-primary">AI Analysis</Link></li>
            </ul>
          </div>
          <div>
            <h4 className="font-semibold mb-4">Resources</h4>
            <ul className="space-y-2 text-sm">
              <li><a href="#" className="text-gray-600 hover:text-interview-primary">Interview Tips</a></li>
              <li><a href="#" className="text-gray-600 hover:text-interview-primary">Career Advice</a></li>
              <li><a href="#" className="text-gray-600 hover:text-interview-primary">FAQ</a></li>
            </ul>
          </div>
          <div>
            <h4 className="font-semibold mb-4">Company</h4>
            <ul className="space-y-2 text-sm">
              <li><a href="#" className="text-gray-600 hover:text-interview-primary">About Us</a></li>
              <li><a href="#" className="text-gray-600 hover:text-interview-primary">Contact</a></li>
              <li><a href="#" className="text-gray-600 hover:text-interview-primary">Privacy Policy</a></li>
            </ul>
          </div>
        </div>
        <div className="border-t mt-8 pt-8 text-center text-sm text-gray-600">
          <p>© {new Date().getFullYear()} InterviewCoach. All rights reserved.</p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
