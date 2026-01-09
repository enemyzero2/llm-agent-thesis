# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a bachelor's thesis project (本科毕业设计) from Beijing University of Posts and Telecommunications.

**Title:** 基于大语言模型的智能驱动系统的设计与实现 (Design and Implementation of an Intelligent Driven System Based on Large Language Models)

**Status:** Proposal/Design Phase - no implementation code yet

## Research Focus

The project aims to build an intelligent automation system combining:
- Large Language Models (LLM) for natural language understanding
- Intent-Based Networking (IBN) for network management
- Multi-Agent collaboration using CrewAI framework
- Model Context Protocol (MCP) for tool integration
- Application domain: "New Calling" (新通话) telecommunication service

## Planned Architecture

Key components from the proposal:
- **Planner-Executor Architecture** - Hierarchical design replacing single-model intent classification
- **RCI (Recursive Criticism and Improvement)** - Self-reflection and correction mechanism
- **MCP Integration** - Standard protocol layer connecting models to external tools
- **Multi-Agent System** - CrewAI-based agent coordination
- **Mock Feedback Environment** - For testing and validation

## Repository Contents

Current files are research materials:
- `开题报告-侯林宝.docx` - Thesis proposal document
- `周报.docx` - Weekly progress report
- Research literature PDFs (10 papers on LLM agents, IBN, GUI automation)

## When Implementation Begins

Expected technology stack based on proposal:
- Python (likely primary language for LLM integration)
- CrewAI framework for multi-agent orchestration
- MCP server implementations for communication services

Future CLAUDE.md updates should include:
- Build/test/lint commands once project structure is established
- Source code architecture and key modules
- Development workflow and testing procedures
