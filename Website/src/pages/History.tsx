
import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import Header from '@/components/layout/Header';
import Footer from '@/components/layout/Footer';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  LineChart,
  Line
} from 'recharts';
import { Calendar } from '@/components/ui/calendar';
import { Button } from '@/components/ui/button';
import { ChevronRight } from 'lucide-react';
import { Link } from 'react-router-dom';

// Mock data for the history page
const mockInterviewData = [
  {
    id: 1,
    date: '2025-04-09',
    type: 'Behavioral',
    scores: {
      overall: 78,
      confidence: 81,
      engagement: 76,
      communication: 72,
      bodyLanguage: 80,
      eyeContact: 85
    }
  },
  {
    id: 2,
    date: '2025-04-06',
    type: 'Technical',
    scores: {
      overall: 72,
      confidence: 75,
      engagement: 70,
      communication: 68,
      bodyLanguage: 73,
      eyeContact: 79
    }
  },
  {
    id: 3,
    date: '2025-04-03',
    type: 'Behavioral',
    scores: {
      overall: 65,
      confidence: 62,
      engagement: 68,
      communication: 63,
      bodyLanguage: 67,
      eyeContact: 70
    }
  },
  {
    id: 4,
    date: '2025-03-29',
    type: 'Situational',
    scores: {
      overall: 61,
      confidence: 58,
      engagement: 63,
      communication: 59,
      bodyLanguage: 64,
      eyeContact: 65
    }
  }
];

// Progress data for charts
const progressData = [
  { date: 'Mar 29', score: 61 },
  { date: 'Apr 03', score: 65 },
  { date: 'Apr 06', score: 72 },
  { date: 'Apr 09', score: 78 }
];

const skillProgressData = [
  { name: 'Confidence', Mar29: 58, Apr03: 62, Apr06: 75, Apr09: 81 },
  { name: 'Engagement', Mar29: 63, Apr03: 68, Apr06: 70, Apr09: 76 },
  { name: 'Communication', Mar29: 59, Apr03: 63, Apr06: 68, Apr09: 72 },
  { name: 'Body Language', Mar29: 64, Apr03: 67, Apr06: 73, Apr09: 80 },
  { name: 'Eye Contact', Mar29: 65, Apr03: 70, Apr06: 79, Apr09: 85 }
];

const History = () => {
  return (
    <div className="min-h-screen flex flex-col bg-gray-50">
      <Header />
      
      <main className="flex-grow container mx-auto px-4 py-8">
        <h1 className="text-3xl font-bold mb-6">Your Interview History</h1>
        
        <Tabs defaultValue="progress" className="w-full">
          <TabsList className="mb-6">
            <TabsTrigger value="progress">Progress</TabsTrigger>
            <TabsTrigger value="history">Past Interviews</TabsTrigger>
            <TabsTrigger value="calendar">Calendar</TabsTrigger>
          </TabsList>
          
          <TabsContent value="progress">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
              <Card>
                <CardHeader>
                  <CardTitle>Overall Progress</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-72">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart
                        data={progressData}
                        margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="date" />
                        <YAxis domain={[0, 100]} />
                        <Tooltip />
                        <Line 
                          type="monotone" 
                          dataKey="score" 
                          stroke="#3B82F6" 
                          strokeWidth={3} 
                          dot={{ r: 6 }} 
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="mt-4 text-center text-sm text-gray-600">
                    <p>Your overall interview score has improved by 17 points in the last 2 weeks.</p>
                  </div>
                </CardContent>
              </Card>
              
              <Card>
                <CardHeader>
                  <CardTitle>Skills Improvement</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-72">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart
                        data={skillProgressData}
                        margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="name" />
                        <YAxis domain={[0, 100]} />
                        <Tooltip />
                        <Bar dataKey="Mar29" name="Mar 29" fill="#94A3B8" />
                        <Bar dataKey="Apr03" name="Apr 03" fill="#64748B" />
                        <Bar dataKey="Apr06" name="Apr 06" fill="#1E40AF" />
                        <Bar dataKey="Apr09" name="Apr 09" fill="#3B82F6" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="mt-4 text-center text-sm text-gray-600">
                    <p>Eye contact and confidence showed the most improvement.</p>
                  </div>
                </CardContent>
              </Card>
            </div>
            
            <Card>
              <CardHeader>
                <CardTitle>Most Improved Areas</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div className="bg-white p-5 rounded-lg border border-gray-100 text-center">
                    <div className="text-interview-primary text-3xl font-bold mb-2">+23%</div>
                    <div className="text-lg font-medium">Eye Contact</div>
                    <p className="text-sm text-gray-600 mt-2">
                      Significant improvement in maintaining consistent eye contact
                    </p>
                  </div>
                  
                  <div className="bg-white p-5 rounded-lg border border-gray-100 text-center">
                    <div className="text-interview-primary text-3xl font-bold mb-2">+20%</div>
                    <div className="text-lg font-medium">Confidence</div>
                    <p className="text-sm text-gray-600 mt-2">
                      More assured delivery and reduced nervous behaviors
                    </p>
                  </div>
                  
                  <div className="bg-white p-5 rounded-lg border border-gray-100 text-center">
                    <div className="text-interview-primary text-3xl font-bold mb-2">+16%</div>
                    <div className="text-lg font-medium">Body Language</div>
                    <p className="text-sm text-gray-600 mt-2">
                      Better posture and more effective hand gestures
                    </p>
                  </div>
                </div>
                
                <div className="mt-8 text-center">
                  <Button asChild className="bg-interview-primary hover:bg-interview-primary/90">
                    <Link to="/practice">
                      Continue Practicing <ChevronRight className="ml-2 h-4 w-4" />
                    </Link>
                  </Button>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
          
          <TabsContent value="history">
            <div className="space-y-6">
              {mockInterviewData.map((interview) => (
                <Card key={interview.id}>
                  <CardContent className="p-6">
                    <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-4">
                      <div>
                        <h3 className="text-lg font-semibold">{interview.type} Interview</h3>
                        <p className="text-sm text-gray-600">
                          {new Date(interview.date).toLocaleDateString('en-US', { 
                            weekday: 'long', 
                            year: 'numeric', 
                            month: 'long', 
                            day: 'numeric' 
                          })}
                        </p>
                      </div>
                      <div className="flex items-center gap-4 mt-2 md:mt-0">
                        <div className="bg-interview-background text-interview-primary px-4 py-2 rounded-full font-medium">
                          {interview.scores.overall}% Overall
                        </div>
                        <Button variant="outline" size="sm" asChild>
                          <Link to={`/history/${interview.id}`}>View Details</Link>
                        </Button>
                      </div>
                    </div>
                    
                    <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mt-6">
                      <div className="text-center">
                        <div className="text-lg font-semibold">
                          {interview.scores.confidence}%
                        </div>
                        <p className="text-xs text-gray-600">Confidence</p>
                      </div>
                      <div className="text-center">
                        <div className="text-lg font-semibold">
                          {interview.scores.engagement}%
                        </div>
                        <p className="text-xs text-gray-600">Engagement</p>
                      </div>
                      <div className="text-center">
                        <div className="text-lg font-semibold">
                          {interview.scores.communication}%
                        </div>
                        <p className="text-xs text-gray-600">Communication</p>
                      </div>
                      <div className="text-center">
                        <div className="text-lg font-semibold">
                          {interview.scores.bodyLanguage}%
                        </div>
                        <p className="text-xs text-gray-600">Body Language</p>
                      </div>
                      <div className="text-center">
                        <div className="text-lg font-semibold">
                          {interview.scores.eyeContact}%
                        </div>
                        <p className="text-xs text-gray-600">Eye Contact</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
              
              <div className="text-center mt-8">
                <Button variant="outline">Load More History</Button>
              </div>
            </div>
          </TabsContent>
          
          <TabsContent value="calendar">
            <Card>
              <CardHeader>
                <CardTitle>Interview Calendar</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex flex-col items-center md:flex-row md:items-start gap-8">
                  <div className="border rounded-md">
                    <Calendar
                      mode="single"
                      selected={new Date()}
                      className="rounded-md"
                    />
                  </div>
                  
                  <div className="flex-1">
                    <h3 className="text-lg font-semibold mb-4">Upcoming Sessions</h3>
                    <div className="space-y-4">
                      <div className="p-4 border rounded-lg bg-white">
                        <div className="flex justify-between items-start">
                          <div>
                            <p className="font-medium">Mock Technical Interview</p>
                            <p className="text-sm text-gray-600">Tomorrow, 10:00 AM</p>
                          </div>
                          <Button size="sm" variant="outline">Reschedule</Button>
                        </div>
                      </div>
                      
                      <div className="p-4 border rounded-lg bg-white">
                        <div className="flex justify-between items-start">
                          <div>
                            <p className="font-medium">Weekly Progress Review</p>
                            <p className="text-sm text-gray-600">Friday, April 12, 2:00 PM</p>
                          </div>
                          <Button size="sm" variant="outline">Reschedule</Button>
                        </div>
                      </div>
                    </div>
                    
                    <div className="mt-8">
                      <Button className="w-full bg-interview-primary hover:bg-interview-primary/90">
                        Schedule New Practice
                      </Button>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </main>
      
      <Footer />
    </div>
  );
};

export default History;
