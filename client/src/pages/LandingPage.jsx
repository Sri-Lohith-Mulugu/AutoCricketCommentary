import React from "react";
import { useNavigate } from "react-router-dom"; // To navigate between pages
import "./LandingPage.css"; // Make sure your styles are linked

const LandingPage = () => {
  const navigate = useNavigate(); // hook to navigate to different pages

  const handleLogin = () => {
    navigate("/login"); // Navigate to the login page
  };

  const handleSignup = () => {
    navigate("/register"); // Navigate to the signup page
  };

  return (
    <div className="hero-container">
      <div className="hero-buttons">
        <button className="button" onClick={handleLogin}>
          Login
        </button>
        <button className="button" onClick={handleSignup}>
          Register
        </button>
      </div>

      <section className="hero">
        <div className="hero-text">
          <h1>Cricket Commentary Powered by CNN</h1>
          <p>
          Our cutting-edge Convolutional Neural Network (CNN) 
          can analyze cricket videos in real-time and provide detailed commentary. 
          It recognizes key events, player actions, and game highlights, offering insightful commentary to enhance the viewing experience. 
          With deep learning and computer vision technologies, our CNN transforms your cricket viewing into an interactive and immersive experience.
          </p>
        </div>

        <div className="hero-image">
          <img src="/image.png" alt="Cricket Game" />
        </div>
      </section>
    </div>
  );
};

export default LandingPage;
