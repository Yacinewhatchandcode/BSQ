import { registerRootComponent } from 'expo';
import React, { useState, useEffect, useRef } from 'react';
import { 
  StyleSheet, 
  Text, 
  View, 
  TextInput, 
  TouchableOpacity, 
  ScrollView, 
  KeyboardAvoidingView, 
  Platform,
  Animated,
  Dimensions,
  ActivityIndicator,
  Alert
} from 'react-native';
import { Audio } from 'expo-av';
import { MaterialIcons } from '@expo/vector-icons';
import * as FileSystem from 'expo-file-system';
import * as Speech from 'expo-speech';

const { width } = Dimensions.get('window');

export default function App() {
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [recording, setRecording] = useState(null);
  const ws = useRef(null);
  const scrollViewRef = useRef(null);
  const fadeAnim = useRef(new Animated.Value(0)).current;
  const slideAnim = useRef(new Animated.Value(50)).current;

  useEffect(() => {
    // Start animation
    Animated.parallel([
      Animated.timing(fadeAnim, {
        toValue: 1,
        duration: 1000,
        useNativeDriver: true,
      }),
      Animated.timing(slideAnim, {
        toValue: 0,
        duration: 1000,
        useNativeDriver: true,
      }),
    ]).start();

    // WebSocket setup
    const connectWebSocket = () => {
      // Use localhost for development
      const wsUrl = Platform.OS === 'web' ? 'ws://localhost:8000/ws' : 'ws://192.168.1.171:8000/ws';
      console.log('Connecting to WebSocket:', wsUrl);
      ws.current = new WebSocket(wsUrl);

      ws.current.onopen = () => {
        console.log('WebSocket Connected');
        Alert.alert('Connected', 'Successfully connected to the server');
      };

      ws.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'response') {
            setIsTyping(false);
            setMessages(prev => [...prev, { type: 'bot', content: data.content }]);
            Speech.speak(data.content);
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      ws.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        Alert.alert(
          'Connection Error',
          'Failed to connect to the server. Please check if the server is running and try again.',
          [
            {
              text: 'Retry',
              onPress: () => {
                setTimeout(connectWebSocket, 3000);
              }
            }
          ]
        );
      };

      ws.current.onclose = () => {
        console.log('WebSocket disconnected. Reconnecting...');
        setTimeout(connectWebSocket, 3000);
      };
    };

    connectWebSocket();

    return () => {
      if (ws.current) {
        ws.current.close();
      }
    };
  }, []);

  const startRecording = async () => {
    try {
      const { status } = await Audio.requestPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert('Permission required', 'Please grant microphone permissions.');
        return;
      }
      await Audio.setAudioModeAsync({ allowsRecordingIOS: true, playsInSilentModeIOS: true });
      const newRecording = new Audio.Recording();
      await newRecording.prepareToRecordAsync(Audio.RecordingOptionsPresets.HIGH_QUALITY);
      await newRecording.startAsync();
      setRecording(newRecording);
      setIsRecording(true);
      Alert.alert('Recording started', 'Hold the button to record.');
    } catch (error) {
      setIsRecording(false);
      setRecording(null);
      Alert.alert('Error', 'Failed to start recording.');
    }
  };

  const stopRecording = async () => {
    if (!recording) {
      Alert.alert('Error', 'No recording in progress.');
      return;
    }
    try {
      await recording.stopAndUnloadAsync();
      setIsRecording(false);
      const uri = recording.getURI();
      setRecording(null);

      if (!uri) {
        Alert.alert('Error', 'No audio file was created.');
        return;
      }

      // Check file size
      const fileInfo = await FileSystem.getInfoAsync(uri);
      if (!fileInfo.exists || fileInfo.size < 1000) { // 1KB minimum
        Alert.alert('Error', 'Recording failed or is too short.');
        return;
      }

      // Now upload the file
      const formData = new FormData();
      formData.append('file', {
        uri,
        type: 'audio/m4a',
        name: 'recording.m4a'
      });

      const serverUrl = Platform.OS === 'web'
        ? 'http://localhost:8000/transcribe'
        : 'http://192.168.1.171:8000/transcribe';

      const response = await fetch(serverUrl, {
        method: 'POST',
        body: formData,
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      if (data.text) {
        sendMessage(data.text);
      } else {
        Alert.alert('Error', 'Failed to transcribe audio. Please try again.');
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to stop or upload recording.');
    }
  };

  const sendMessage = () => {
    if (inputText.trim() && ws.current && ws.current.readyState === WebSocket.OPEN) {
      setMessages(prev => [...prev, { type: 'user', content: inputText }]);
      ws.current.send(JSON.stringify({ type: 'message', content: inputText }));
      setInputText('');
      setIsTyping(true);
    } else {
      Alert.alert('Connection Error', 'Cannot send message. Please check your connection.');
    }
  };

  const renderMessage = (message, index) => {
    const isQuote = message.content.includes('O ') && message.content.includes('!');
    
    return (
      <Animated.View
        key={index}
        style={[
          styles.messageBubble,
          message.type === 'user' ? styles.userMessage : styles.botMessage,
          {
            opacity: fadeAnim,
            transform: [{ translateY: slideAnim }],
          },
        ]}
      >
        {isQuote ? (
          <View style={styles.quoteContainer}>
            <Text style={styles.quoteText}>{message.content}</Text>
          </View>
        ) : (
          <Text style={message.type === 'user' ? styles.userMessageText : styles.botMessageText}>
            {message.content}
          </Text>
        )}
      </Animated.View>
    );
  };

  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
      keyboardVerticalOffset={Platform.OS === 'ios' ? 90 : 0}
    >
      <Animated.View 
        style={[
          styles.header,
          {
            opacity: fadeAnim,
            transform: [{ translateY: slideAnim }],
          },
        ]}
      >
        <Text style={styles.title}>Spiritual Quest</Text>
      </Animated.View>

      <Animated.View 
        style={[
          styles.inputContainer,
          {
            opacity: fadeAnim,
            transform: [{ translateY: slideAnim }],
          },
        ]}
      >
        <TextInput
          style={styles.input}
          value={inputText}
          onChangeText={setInputText}
          placeholder="Ask for wisdom or share your feelings..."
          placeholderTextColor="#666"
          multiline
        />
        {Platform.OS !== 'web' && (
          <TouchableOpacity 
            style={[styles.micButton, isRecording && styles.micButtonActive]} 
            onPressIn={startRecording}
            onPressOut={stopRecording}
          >
            <MaterialIcons 
              name={isRecording ? "mic" : "mic-none"} 
              size={24} 
              color={isRecording ? "#fff" : "#667eea"} 
            />
          </TouchableOpacity>
        )}
        <TouchableOpacity 
          style={styles.sendButton} 
          onPress={sendMessage}
          disabled={!inputText.trim()}
        >
          <MaterialIcons name="send" size={24} color="white" />
        </TouchableOpacity>
      </Animated.View>

      <ScrollView
        ref={scrollViewRef}
        style={styles.messagesContainer}
        contentContainerStyle={styles.messagesContentContainer}
        onContentSizeChange={() => scrollViewRef.current.scrollToEnd({ animated: true })}
      >
        {messages.map(renderMessage)}
        {isTyping && (
          <View style={[styles.messageBubble, styles.botMessage]}>
            <ActivityIndicator size="small" color="#667eea" />
          </View>
        )}
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f8fafc',
  },
  header: {
    padding: 20,
    backgroundColor: 'white',
    alignItems: 'center',
    borderBottomWidth: 1,
    borderBottomColor: '#e5e7eb',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  title: {
    fontSize: 24, // Reduced size
    fontWeight: 'bold',
    color: '#1f2937',
  },
  inputContainer: {
    flexDirection: 'row',
    padding: 12,
    backgroundColor: 'white',
    borderBottomWidth: 1,
    borderBottomColor: '#e5e7eb',
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 5,
  },
  input: {
    flex: 1,
    backgroundColor: '#f3f4f6',
    borderRadius: 20,
    paddingHorizontal: 16,
    paddingVertical: 8,
    marginRight: 8,
    color: '#1f2937',
    maxHeight: 100,
    fontSize: 16,
  },
  messagesContainer: {
    flex: 1,
  },
  messagesContentContainer: {
    padding: 16,
    paddingBottom: 32,
  },
  messageBubble: {
    maxWidth: width * 0.75,
    marginVertical: 4,
    padding: 12,
    borderRadius: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
    elevation: 2,
  },
  userMessage: {
    backgroundColor: '#667eea',
    alignSelf: 'flex-end',
    borderBottomRightRadius: 4,
  },
  botMessage: {
    backgroundColor: 'white',
    alignSelf: 'flex-start',
    borderBottomLeftRadius: 4,
    backgroundColor: '#f1f1f1', // Added background for bot messages
  },
  userMessageText: {
    color: 'white',
    fontSize: 16,
  },
  botMessageText: {
    color: '#1f2937',
    fontSize: 16,
  },
  quoteContainer: {
    backgroundColor: 'white',
    padding: 15,
    borderRadius: 12,
    borderLeftWidth: 4,
    borderLeftColor: '#667eea',
  },
  quoteText: {
    fontFamily: Platform.OS === 'ios' ? 'Georgia' : 'serif', // Changed font
    fontSize: 20,
    lineHeight: 28,
    color: '#1f2937',
  },
  micButton: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: '#f3f4f6',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 8,
  },
  micButtonActive: {
    backgroundColor: '#667eea',
  },
  sendButton: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: '#667eea',
    justifyContent: 'center',
    alignItems: 'center',
  },
});

// Register the app
registerRootComponent(App);
 