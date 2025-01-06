import React from "react";
import "./Navbar.css"; // Ensure this file contains the CSS for the navbar
// import logout from "./Logout";
const Navbar = () => {
  return (
    <div className="html-container">
      <div className="html-section">
        <div className="html-project">
          <div className="navigation">
            <nav>
              <ul className="nav-type">
                <li>
                  <a href="/home" className="active">
                    Home
                  </a>
                </li>
                <li>
                  <a href="/about" className="active1">
                    About
                  </a>
                </li>
                <li>
                  <a href="/contact" className="active2">
                    Contact
                  </a>
                </li>
                <li>
                  <a href="/" className="active3">
                    Logout
                  </a>
                </li>
              </ul>
            </nav>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Navbar;
