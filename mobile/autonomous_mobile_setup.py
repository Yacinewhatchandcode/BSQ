#!/usr/bin/env python3
"""
📱 BAHA'I MOBILE AUTONOMOUS SETUP
Auto-configures React Native/Expo mobile app for the Baha'i Spiritual Quest
"""

import os
import json
import subprocess
import sys
from pathlib import Path

def create_mobile_app():
    """Create Expo mobile app with Baha'i theme"""
    
    mobile_dir = Path(__file__).parent
    
    # Create package.json for Expo app
    package_json = {
        "name": "bahai-spiritual-quest",
        "version": "1.0.0",
        "main": "node_modules/expo/AppEntry.js",
        "scripts": {
            "start": "expo start",
            "android": "expo start --android", 
            "ios": "expo start --ios",
            "web": "expo start --web"
        },
        "dependencies": {
            "expo": "~49.0.0",
            "react": "18.2.0",
            "react-native": "0.72.6",
            "react-native-web": "~0.19.6",
            "expo-status-bar": "~1.6.0",
            "expo-font": "~11.4.0",
            "expo-linear-gradient": "~12.3.0",
            "expo-av": "~13.4.0",
            "react-native-paper": "^5.10.0",
            "react-native-vector-icons": "^10.0.0"
        },
        "devDependencies": {
            "@babel/core": "^7.20.0"
        }
    }
    
    with open(mobile_dir / "package.json", "w") as f:
        json.dump(package_json, f, indent=2)
    
    # Create App.js - Main Baha'i mobile interface
    app_js = '''import React, { useState, useEffect } from 'react';
import {
  StyleSheet,
  Text,
  View,
  TextInput,
  TouchableOpacity,
  ScrollView,
  StatusBar,
  Dimensions
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';

const { width, height } = Dimensions.get('window');

export default function App() {
  const [message, setMessage] = useState('');
  const [messages, setMessages] = useState([]);
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    // Connect to backend WebSocket
    connectToBackend();
  }, []);

  const connectToBackend = () => {
    try {
      const ws = new WebSocket('ws://localhost:8000/ws');
      
      ws.onopen = () => {
        setIsConnected(true);
        console.log('Connected to Baha\\'i backend');
      };
      
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'response') {
          addMessage(data.content, 'agent');
        }
      };
      
      ws.onclose = () => {
        setIsConnected(false);
        setTimeout(connectToBackend, 5000); // Reconnect
      };
      
    } catch (error) {
      console.log('WebSocket connection failed, using HTTP');
      setIsConnected(false);
    }
  };

  const addMessage = (text, sender) => {
    const newMessage = {
      id: Date.now(),
      text,
      sender,
      timestamp: new Date().toLocaleTimeString()
    };
    setMessages(prev => [...prev, newMessage]);
  };

  const sendMessage = async () => {
    if (!message.trim()) return;
    
    addMessage(message, 'user');
    const currentMessage = message;
    setMessage('');
    
    try {
      // Try HTTP API as fallback
      const response = await fetch('http://localhost:8000/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: `message=${encodeURIComponent(currentMessage)}`,
      });
      
      const data = await response.json();
      addMessage(data.response, 'agent');
    } catch (error) {
      addMessage('Connection error. Please check if the backend server is running.', 'agent');
    }
  };

  const renderMessage = (msg) => {
    const isUser = msg.sender === 'user';
    return (
      <View
        key={msg.id}
        style={[
          styles.messageContainer,
          isUser ? styles.userMessage : styles.agentMessage
        ]}
      >
        <Text style={[
          styles.messageText,
          isUser ? styles.userText : styles.agentText
        ]}>
          {msg.text}
        </Text>
        <Text style={styles.timestamp}>{msg.timestamp}</Text>
      </View>
    );
  };

  return (
    <LinearGradient
      colors={['#0B0E1A', '#1A1F2E', '#2D3A4B']}
      style={styles.container}
    >
      <StatusBar barStyle="light-content" />
      
      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.titlePersian}>كلمات مخفیه</Text>
        <Text style={styles.titleEnglish}>The Hidden Words</Text>
        <View style={[
          styles.connectionStatus,
          { backgroundColor: isConnected ? '#28a745' : '#dc3545' }
        ]}>
          <Text style={styles.statusText}>
            {isConnected ? 'Connected' : 'Offline'}
          </Text>
        </View>
      </View>

      {/* Messages */}
      <ScrollView 
        style={styles.messagesContainer}
        contentContainerStyle={styles.messagesContent}
      >
        {messages.length === 0 ? (
          <View style={styles.welcomeContainer}>
            <Text style={styles.welcomeText}>
              Welcome to this sacred digital space, where the eternal wisdom of{' '}
              <Text style={styles.emphasis}>The Hidden Words</Text> illuminates our spiritual journey.
            </Text>
          </View>
        ) : (
          messages.map(renderMessage)
        )}
      </ScrollView>

      {/* Input */}
      <View style={styles.inputContainer}>
        <TextInput
          style={styles.textInput}
          value={message}
          onChangeText={setMessage}
          placeholder="Seek wisdom from The Hidden Words..."
          placeholderTextColor="#666"
          multiline
          onSubmitEditing={sendMessage}
        />
        <TouchableOpacity
          style={styles.sendButton}
          onPress={sendMessage}
        >
          <Text style={styles.sendButtonText}>REVEAL</Text>
        </TouchableOpacity>
      </View>
    </LinearGradient>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    paddingTop: StatusBar.currentHeight || 40,
  },
  header: {
    alignItems: 'center',
    paddingVertical: 20,
    paddingHorizontal: 20,
  },
  titlePersian: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#E6D4A3',
    marginBottom: 5,
    textAlign: 'center',
  },
  titleEnglish: {
    fontSize: 14,
    color: '#A8A8A8',
    letterSpacing: 2,
    textTransform: 'uppercase',
    marginBottom: 10,
  },
  connectionStatus: {
    paddingHorizontal: 12,
    paddingVertical: 4,
    borderRadius: 12,
  },
  statusText: {
    color: 'white',
    fontSize: 12,
    fontWeight: '600',
  },
  messagesContainer: {
    flex: 1,
    paddingHorizontal: 20,
  },
  messagesContent: {
    paddingVertical: 10,
  },
  welcomeContainer: {
    padding: 20,
    marginVertical: 20,
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
    borderRadius: 15,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.1)',
  },
  welcomeText: {
    color: '#CCCCCC',
    fontSize: 16,
    lineHeight: 24,
    textAlign: 'center',
  },
  emphasis: {
    color: '#E6D4A3',
    fontStyle: 'italic',
  },
  messageContainer: {
    marginVertical: 8,
    padding: 15,
    borderRadius: 15,
    maxWidth: width * 0.8,
  },
  userMessage: {
    alignSelf: 'flex-end',
    backgroundColor: 'rgba(212, 175, 55, 0.2)',
    borderColor: 'rgba(212, 175, 55, 0.3)',
    borderWidth: 1,
  },
  agentMessage: {
    alignSelf: 'flex-start',
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
    borderColor: 'rgba(255, 255, 255, 0.1)',
    borderWidth: 1,
  },
  messageText: {
    fontSize: 16,
    lineHeight: 22,
  },
  userText: {
    color: '#E0E0E0',
  },
  agentText: {
    color: '#F5E6C8',
    fontStyle: 'italic',
  },
  timestamp: {
    fontSize: 12,
    color: '#888',
    marginTop: 5,
    textAlign: 'right',
  },
  inputContainer: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    paddingHorizontal: 20,
    paddingVertical: 20,
    paddingBottom: 30,
    backgroundColor: 'rgba(0, 0, 0, 0.3)',
  },
  textInput: {
    flex: 1,
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    borderColor: 'rgba(255, 255, 255, 0.2)',
    borderWidth: 1,
    borderRadius: 25,
    paddingHorizontal: 20,
    paddingVertical: 15,
    color: '#CCCCCC',
    fontSize: 16,
    maxHeight: 100,
    marginRight: 10,
  },
  sendButton: {
    backgroundColor: '#D4AF37',
    paddingHorizontal: 25,
    paddingVertical: 15,
    borderRadius: 25,
    shadowColor: '#D4AF37',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.3,
    shadowRadius: 4,
    elevation: 5,
  },
  sendButtonText: {
    color: '#1A1A1A',
    fontWeight: '600',
    fontSize: 14,
    letterSpacing: 1,
  },
});'''
    
    with open(mobile_dir / "App.js", "w") as f:
        f.write(app_js)
    
    # Create app.json for Expo
    app_json = {
        "expo": {
            "name": "Bahá'í Spiritual Quest",
            "slug": "bahai-spiritual-quest", 
            "version": "1.0.0",
            "orientation": "portrait",
            "icon": "./assets/icon.png",
            "userInterfaceStyle": "dark",
            "splash": {
                "image": "./assets/splash.png",
                "resizeMode": "contain",
                "backgroundColor": "#0B0E1A"
            },
            "updates": {
                "fallbackToCacheTimeout": 0
            },
            "assetBundlePatterns": ["**/*"],
            "ios": {
                "supportsTablet": True
            },
            "android": {
                "adaptiveIcon": {
                    "foregroundImage": "./assets/adaptive-icon.png",
                    "backgroundColor": "#0B0E1A"
                }
            },
            "web": {
                "favicon": "./assets/favicon.png"
            }
        }
    }
    
    with open(mobile_dir / "app.json", "w") as f:
        json.dump(app_json, f, indent=2)
    
    print("📱 Mobile app structure created successfully!")
    print("🌟 Bahá'í Spiritual Quest mobile app ready for launch")

if __name__ == "__main__":
    create_mobile_app()