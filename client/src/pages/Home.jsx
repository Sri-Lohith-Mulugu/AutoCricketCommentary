// src/pages/Home.js
import React from "react";
import Navbar from "./Navbar"; // Assuming you have a Navbar component
import Card from "./Card"; // Assuming you have a Card component

const Home = () => {
  return (
    <div className="home-container">
      <Navbar /> {/* Render the Navbar at the top */}
      <div className="card-container">
        <Card /> {/* Render the Card centered */}
      </div>
    </div>
  );
};

export default Home;
