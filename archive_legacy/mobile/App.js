import React, { useState, useEffect } from 'react';
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
        console.log('Connected to Baha\'i backend');
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
});