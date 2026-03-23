import serial
import struct
import time
import tkinter as tk
from tkinter import messagebox
import serial.tools.list_ports

class EspCli:
    """
    Module to communicate with the WALL-E Master via a binary protocol.
    Matches the structure and IDs from Protocol.h
    Protocol: [Cmd-Id (1B)] [Target-Id (1B)] [Len (1B)] [Payload (max 29B)]
    """
    
    # Command IDs from  Protocol.h / Commands.h
    # !! DO NOT CHANGE !!
    CHANGE_STATE   = 3
    UPDATE_GROUP   = 6
    UPDATE_WIFI    = 10
    UPDATE_URL     = 11
    UPDATE_POS     = 16
    UPDATE_EYE     = 17
    TEST_LED       = 18
    UPDATE_ANGLE   = 20
    TEST_SERVO     = 21
    UPDATE_FW      = 22

    BROADCAST_ID   = 255   

    def __init__(self, port, baudrate=115200, timeout=1):
        """Initializes the serial connection to the Master ESP32."""
        try:
            self.ser = serial.Serial(port, baudrate, timeout=timeout)
            # Short delay to allow ESP32 to reboot if DTR/RTS is toggled
            time.sleep(1.5) 
            print(f"Connected to WALL-E Master on {port}")
        except Exception as e:
            print(f"Error opening serial port: {e}")

    def _send_packet(self, cmd_id, target_id, payload_bytes):
        """Internal helper to pack and send the binary frame."""
        length = len(payload_bytes)
        if length > 29:
            print(f"Warning: Payload too long ({length} bytes). Max is 29.")
            payload_bytes = payload_bytes[:29]
            length = 29
            
        # Header: Cmd (B), Target (B), Len (B)
        header = struct.pack("BBB", cmd_id, target_id, length)
        packet = header + payload_bytes
        self.ser.write(packet)
        self.ser.flush()

    def change_state(self, target_id, state_str):
        """Sends a state change command (e.g., 'I', 'M' or 'A')."""
        payload = state_str.encode('ascii')[:29]
        self._send_packet(self.CHANGE_STATE, target_id, payload)

    def update_group(self, target_id, group_id):
        """Sets the group ID for a specific target (int)."""
        payload = struct.pack("i", group_id)
        self._send_packet(self.UPDATE_GROUP, target_id, payload)

    def update_pos(self, target_id, x, y, z):
        """
        Sends 3D coordinates. 
        Matches 'CoordinatesPayload' struct (float x, y, z).
        """
        payload = struct.pack("fff", x, y, z)
        self._send_packet(self.UPDATE_POS, target_id, payload)

    def update_eye(self, target_id, pattern_str):
        """Sends a pattern string for the LED eyes."""
        payload = pattern_str.encode('ascii')[:29]
        self._send_packet(self.UPDATE_EYE, target_id, payload)

    def test_led(self, target_id=BROADCAST_ID):
        """Triggers the LED test sequence."""
        self._send_packet(self.TEST_LED, target_id, b"")

    def update_angle(self, target_id, yaw, tilt):
        """
        Sends servo angles. 
        Matches 'AnglePayload' struct (float yaw, tilt).
        """
        payload = struct.pack("<ff", yaw, tilt)
        self._send_packet(self.UPDATE_ANGLE, target_id, payload)

    def test_servo(self, target_id=BROADCAST_ID):
        """Triggers the Servo test sequence."""
        self._send_packet(self.TEST_SERVO, target_id, b"")

    def update_fw(self, target_id, version_str="teststr"):
        """Sends the command to trigger OTA update."""
        payload = version_str.encode('ascii')[:29]
        self._send_packet(self.UPDATE_FW, target_id, payload)

    def update_url(self, target_id, url_str):
        """Updates the OTA base URL on the robot."""
        payload = url_str.encode('ascii')[:29]
        self._send_packet(self.UPDATE_URL, target_id, payload)

    def update_wifi(self, target_id, ssid, pwd):
        """
        Updates the Wi-Fi credentials on the robot.
        Payload: ssid[10] (10 bytes), pwd[19] (19 bytes) = 29 bytes total.
        """
        # Wir kodieren die Strings in Bytes und füllen sie mit Nullbytes (\x00) auf
        # oder schneiden sie ab, falls sie zu lang sind.
        payload = struct.pack("10s19s", ssid.encode('ascii')[:10], pwd.encode('ascii')[:19])
        print(payload)
        self._send_packet(self.UPDATE_WIFI, target_id, payload)

    def close(self):
        """Closes the serial connection."""
        self.ser.close()



class EspConfig:
    """
    Module for configuring WALL-E Clients via Serial String Interface.
    Used for one-time setup (ID, Matrix, WiFi) rather than real-time control.
    """

    def __init__(self, port, baudrate=115200, timeout=1):
        """Initializes the serial connection to a WALL-E Client."""
        try:
            # We don't use DTR/RTS to prevent unwanted reboots during config if possible
            self.ser = serial.Serial(port, baudrate, timeout=timeout)
            time.sleep(2) # Wait for potential bootloader/reboot sequence
            print(f"EspConfig: Connected to Client on {port}")
        except Exception as e:
            print(f"EspConfig Error: {e}")

    def get_robot_id(self):
        """
        Sends 'get_id' and parses the response.
        Expected format: '> Robot ID: 10'
        """
        try:
            self.ser.flushInput()
            self.ser.write(b"get_id\n")
            
            # Read lines until we find the ID pattern or timeout
            start_time = time.time()
            while (time.time() - start_time) < 2:
                line = self.ser.readline().decode('ascii', errors='ignore').strip()
                if "Robot ID:" in line:
                    # Extract the number from the string
                    parts = line.split(":")
                    if len(parts) > 1:
                        robot_id = int(parts[1].strip())
                        return robot_id
            return None
        except Exception as e:
            print(f"Error getting ID: {e}")
            return None

    def upload_matrix(self, matrix_array):
        """
        Enters Matrix mode and uploads 12 float values.
        Waits for ACK (0x06) before sending each value.
        """
        if len(matrix_array) != 12:
            print("Error: Matrix array must contain exactly 12 values.")
            return False

        try:
            print("Starting Matrix Upload Wizard...")
            self.ser.flushInput()
            
            self.ser.write(b"chst M\n") # changing to maintenance mode 
            time.sleep(0.1)
            
            self.ser.write(b"set_transM\n")
            
            val_idx = 0
            while val_idx < 12:
                # Read 1 byte and check for ACK (0x06)
                char = self.ser.read(1)
                
                if char == b'\x06':
                    # Send current float as string
                    val_str = f"{matrix_array[val_idx]}\n"
                    self.ser.write(val_str.encode('ascii'))
                    print(f"Sent [{val_idx+1}/12]: {matrix_array[val_idx]}")
                    
                    val_idx += 1
                    time.sleep(0.05) # Small safety delay
                
            # Wait for final success message from ESP32
            end_time = time.time()
            while (time.time() - end_time) < 3:
                line = self.ser.readline().decode('ascii', errors='ignore').strip()
                if "SAVE_SUCCESS" in line:
                    print("Matrix successfully saved to NVS.")
                    return True
            
            print("Warning: Timed out waiting for SAVE_SUCCESS.")
            return False

        except Exception as e:
            print(f"Matrix Upload Error: {e}")
            return False

    def close(self):
        """Closes the serial connection."""
        if self.ser.is_open:
            self.ser.close()