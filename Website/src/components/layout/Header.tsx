
import React from 'react';
import { Button } from '@/components/ui/button';
import { Link } from 'react-router-dom';

const Header = () => {
  return (
    <header className="border-b bg-white/80 backdrop-blur-sm sticky top-0 z-50">
      <div className="container mx-auto flex h-16 items-center justify-between px-4">
        <div className="flex items-center gap-2">
          <Link to="/" className="flex items-center gap-2">
            <span className="text-2xl font-bold gradient-text">InterviewCoach</span>
          </Link>
        </div>
        <nav className="hidden md:flex items-center gap-6">
          <Link to="/" className="text-sm font-medium hover:text-interview-primary">
            Home
          </Link>
          <Link to="/practice" className="text-sm font-medium hover:text-interview-primary">
            Practice
          </Link>
          <Link to="/history" className="text-sm font-medium hover:text-interview-primary">
            History
          </Link>
          <Link to="/about" className="text-sm font-medium hover:text-interview-primary">
            About
          </Link>
        </nav>
        <div className="flex items-center gap-4">
          <Button variant="outline" size="sm" className="hidden md:flex">
            Sign In
          </Button>
          <Button size="sm" className="bg-interview-primary hover:bg-interview-primary/90">
            Get Started
          </Button>
        </div>
      </div>
    </header>
  );
};

export default Header;
