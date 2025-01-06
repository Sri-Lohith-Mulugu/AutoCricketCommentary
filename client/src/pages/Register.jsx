import React, { useState } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";
import { ToastContainer, toast } from "react-toastify";
import "./style.css";

const Register = () => {
  const [values, setValues] = useState({ email: "", password: "" });
  const navigate = useNavigate(); // Initialize navigate hook

  const generateError = (error) =>
    toast.error(error, {
      position: "bottom-right",
    });

  const handleSubmit = async (event) => {
    event.preventDefault();
    try {
      const { data } = await axios.post("http://localhost:4000/register", {
        email: values.email,
        password: values.password,
      });
      if (data) {
        if (data.errors) {
          const { email, password } = data.errors;
          if (email) generateError(email);
          else if (password) generateError(password);
        } else {
          localStorage.setItem("jwt", data.token); // Store JWT in localStorage
          navigate("/home"); // Redirect to home page after successful registration
        }
      }
    } catch (ex) {
      console.log(ex);
    }
  };

  return (
    <div className="form-container">
      <form onSubmit={handleSubmit} className="register-form">
        <label>Sign Up</label>
        <input
          type="email"
          name="email"
          placeholder="Email"
          onChange={(e) =>
            setValues({ ...values, [e.target.name]: e.target.value })
          }
        />
        <input
          type="password"
          name="password"
          placeholder="Password"
          onChange={(e) =>
            setValues({ ...values, [e.target.name]: e.target.value })
          }
        />
        <button type="submit">Sign Up</button>
        <p>
          Already have an account?{" "}
          <span onClick={() => navigate("/login")}>Login</span>{" "}
          {/* Redirect to login page */}
        </p>
      </form>
      <ToastContainer />
    </div>
  );
};

export default Register;
