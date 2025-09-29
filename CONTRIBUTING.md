# Contributing to LIT for Voice

Thank you for your interest in contributing to LIT for Voice! This document provides guidelines and instructions to help you get started with contributing to this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Setting Up the Development Environment](#setting-up-the-development-environment)
- [Development Workflow](#development-workflow)
  - [Branching Strategy](#branching-strategy)
  - [Making Changes](#making-changes)
  - [Testing](#testing)
  - [Code Style and Linting](#code-style-and-linting)
- [Pull Request Process](#pull-request-process)
- [Documentation](#documentation)
- [Issue Reporting](#issue-reporting)
- [Feature Requests](#feature-requests)

## Code of Conduct

This project follows our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## Getting Started

### Prerequisites

- **Frontend Development**: 
  - Node.js (v18 or higher)
  - npm or bun package manager
  
- **Backend Development**:
  - Python 3.11
  - Docker (for Redis)

### Setting Up the Development Environment

1. **Fork and clone the repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/LIT-for-Voice.git
   cd LIT-for-Voice
   ```

2. **Set up the Frontend**:
   ```bash
   cd Frontend
   npm install
   ```

3. **Set up the Backend**:
   ```bash
   cd Backend
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   source .venv/bin/activate  # On Unix or MacOS
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Start Redis via Docker**:
   ```bash
   cd Backend
   docker compose up -d
   ```

5. **Run the development servers**:
   
   In one terminal (Frontend):
   ```bash
   cd Frontend
   npm run dev
   ```
   
   In another terminal (Backend):
   ```bash
   cd Backend
   .venv\Scripts\activate  # On Windows
   source .venv/bin/activate  # On Unix or MacOS
   uvicorn app.main:app --reload
   ```

## Development Workflow

### Branching Strategy

- `main` is the primary branch and should always be stable
- Create feature branches from `main` using the following naming convention:
  - `feature/short-description` for new features
  - `bugfix/issue-number` for bug fixes
  - `docs/description` for documentation changes
  - `refactor/description` for code refactoring

### Making Changes

1. Create a new branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit them with descriptive messages:
   ```bash
   git add .
   git commit -m "Add detailed description of changes"
   ```

3. Push your branch to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

### Testing

- **Frontend**: We use Jest for testing React components. Run tests with:
  ```bash
  cd Frontend
  npm run test
  ```

- **Backend**: We use pytest for testing API endpoints and services. Run tests with:
  ```bash
  cd Backend
  pytest
  ```

### Code Style and Linting

- **Frontend**: We use ESLint with TypeScript configuration. Run linting with:
  ```bash
  cd Frontend
  npm run lint
  ```

- **Backend**: We follow PEP 8 guidelines. Consider using tools like `flake8` or `black` for formatting.

## Pull Request Process

1. Ensure your code follows our style guidelines and passes all tests
2. Update documentation if necessary
3. Submit a pull request to the `main` branch with a clear title and description
4. Reference any related issues in your PR description using the keyword "Fixes #issue_number"
5. Wait for code review and address any requested changes
6. After approval, a maintainer will merge your PR

## Documentation

- Update the README.md if you're changing functionality or adding features
- Add comments to your code, especially for complex logic
- Consider creating or updating wiki pages for extensive documentation

## Issue Reporting

When reporting issues, please include:

- A clear and descriptive title
- A detailed description of the issue
- Steps to reproduce the problem
- Expected behavior and actual behavior
- Screenshots if applicable
- Environment details (OS, browser, versions, etc.)

## Feature Requests

We welcome feature requests! Please provide:

- A clear and detailed description of the feature
- The motivation and use cases for the feature
- Any potential implementation ideas you might have
- Mockups or examples if applicable

Thank you for contributing to LIT for Voice!