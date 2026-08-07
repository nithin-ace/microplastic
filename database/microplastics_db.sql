CREATE DATABASE IF NOT EXISTS microplastics_db;
USE microplastics_db;

CREATE TABLE IF NOT EXISTS food_safety_authority (
  id INT AUTO_INCREMENT PRIMARY KEY,
  name VARCHAR(100),
  email VARCHAR(100),
  mobile VARCHAR(20),
  username VARCHAR(50),
  password VARCHAR(50)
);

CREATE TABLE IF NOT EXISTS lab_technician (
  id INT AUTO_INCREMENT PRIMARY KEY,
  name VARCHAR(100),
  email VARCHAR(100),
  mobile VARCHAR(20),
  username VARCHAR(50),
  password VARCHAR(50)
);

INSERT INTO lab_technician (name, email, mobile, username, password)
VALUES ('Vijay', 'vijay@gmail.com', '8929090909', 'vijay', '1234')
ON DUPLICATE KEY UPDATE name = VALUES(name), email = VALUES(email), mobile = VALUES(mobile), password = VALUES(password);

CREATE TABLE IF NOT EXISTS researcher (
  id INT AUTO_INCREMENT PRIMARY KEY,
  name VARCHAR(100),
  email VARCHAR(100),
  mobile VARCHAR(20),
  username VARCHAR(50),
  password VARCHAR(50)
);

CREATE TABLE IF NOT EXISTS users (
  id INT AUTO_INCREMENT PRIMARY KEY,
  name VARCHAR(100),
  email VARCHAR(100),
  mobile VARCHAR(20),
  username VARCHAR(50),
  password VARCHAR(50)
);

INSERT INTO users (name, email, mobile, username, password)
VALUES ('Raj', 'akil@gmail.com', '9876543210', 'raj', '1234')
ON DUPLICATE KEY UPDATE name = VALUES(name), email = VALUES(email), mobile = VALUES(mobile), password = VALUES(password);
