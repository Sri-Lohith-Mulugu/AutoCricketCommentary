// LogoutPage.jsx
import React, { useEffect } from "react";
import { useNavigate } from "react-router-dom";

const Logout = () => {
  const navigate = useNavigate();

  useEffect(() => {
    // Clear any authentication data (if applicable)
    localStorage.removeItem("authToken"); // Or sessionStorage or any other storage you're using

    // Redirect to the login page
    navigate("/login");
  }, [navigate]);

  return (
    <div>
      <h1>Logging you out...</h1>
      <p>Redirecting to the login page...</p>
    </div>
  );
};

export default Logout;
