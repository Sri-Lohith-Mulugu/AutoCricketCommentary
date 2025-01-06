import React from "react";
import Navbar from "./Navbar";

const About = () => {
  return (
    <>
      <Navbar />
      <style>
        {`
          .about-text {
            color: white;
            font-size: 18px;
            font-family: Arial, sans-serif;
            margin-top: 150px; /* Adjust this value to add space below the navbar */
            text-align: left;
          }
          h2 {
            color: red;
            font-size: 24px;
            margin-top: 20px;
          }
          p {
            color: white;
            font-size: 20px;
            line-height: 1.6;
          }
          strong {
            font-weight: bold;
          }
        `}
      </style>
      <div className="about-text">
        <p>Welcome to <strong>AI Sports Commentary</strong> — your go-to platform for generating dynamic commentary for sports videos powered by artificial intelligence. Whether you're a content creator, sports enthusiast, or just someone looking to spice up your game footage, our platform offers an innovative way to add expert-level commentary to your videos without the need for a human commentator.</p>

        <h2>What We Do</h2>
        <p>At <strong>AI Sports Commentary</strong>, we leverage cutting-edge AI technology to analyze sports videos and generate play-by-play commentary that mimics the style and excitement of a live sports broadcast. Our platform is designed to work seamlessly with a wide range of sports, from football and basketball to soccer, tennis, and beyond.</p>

        <h2>How It Works</h2>
        <p><strong>Upload Your Video:</strong> Simply upload your sports video to our platform. It could be a highlight reel or even a training session.</p>
        <p><strong>AI Analysis:</strong> Our advanced AI algorithms process the video, detecting key events like goals, assists, turnovers, tackles, and more.</p>
        <p><strong>Generate Commentary:</strong> Based on the AI's analysis, we create natural, engaging commentary tailored to your video.</p>

        <h2>Why Choose Us?</h2>
        <p><strong>Accuracy:</strong> Our AI is trained on thousands of hours of sports footage to ensure that the commentary matches the events in your video with precision.</p>
        <p><strong>Speed:</strong> Get your commentary generated in just minutes. No more waiting for professional voiceovers or spending hours writing scripts.</p>
        <p><strong>User-Friendly:</strong> The process is simple and intuitive, designed for both beginners and experienced users. You don’t need to be a tech expert to use our platform.</p>
        <p><strong>Cost-Effective:</strong> Our AI-powered commentary is affordable, eliminating the need to hire expensive voiceover talent or broadcast professionals.</p>

        <h2>Who Can Benefit?</h2>
        <p><strong>Sports Teams & Coaches:</strong> Enhance your training videos or highlight reels with professional-grade commentary to motivate your players and showcase their skills.</p>
        <p><strong>Content Creators & Streamers:</strong> Add instant commentary to your recorded sports content without needing a full team of commentators.</p>
        <p><strong>Sports Journalists & Bloggers:</strong> Improve the quality of your video reports and analyses with AI-generated commentary that brings your content to life.</p>
        <p><strong>Fans & Enthusiasts:</strong> Whether you want to relive a memorable match or share your own personal game highlights, our platform helps bring your videos to the next level.</p>

        <h2>The Future of Sports Commentary</h2>
        <p>We’re on a mission to revolutionize how people experience sports media. With the power of AI, we’re making it easier than ever to turn any sports video into a thrilling, professional broadcast. As technology continues to evolve, so will our platform, bringing even more customization options, faster processing times, and smarter commentary features.</p>

        <p>Thank you for choosing <strong>AI Sports Commentary</strong> — where your game deserves the spotlight. Let’s make every play count!</p>
      </div>
    </>
  );
};

export default About;
