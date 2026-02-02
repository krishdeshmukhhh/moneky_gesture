import numpy as np

class GestureDetector:
    def __init__(self, min_confidence=0.7):
        self.min_confidence = min_confidence

    def _is_finger_extended(self, lm, finger_tip_idx, finger_pip_idx):
        # Universal check: Tip Y < PIP Y (assuming upright hand)
        # However, for "Think", finger might be pointed somewhat backwards or sideways if tilted.
        # So we also check Euclidean distance from Wrist? No, simpler.
        # Just check if Tip is NOT curled deeply. 
        # Tip to MCP distance > PIP to MCP distance?
        # Let's stick to the prompt's implied simple logic but relax it for "Think".
        return lm[finger_tip_idx].y < lm[finger_pip_idx].y

    def detect_gesture(self, multi_hand_landmarks, face_landmarks=None):
        """
        Analyze logic to return gesture name.
        Args:
            multi_hand_landmarks: List of Hand landmarks.
            face_landmarks: Single Face landmarks (if detected).
        """
        if not multi_hand_landmarks:
            return None

        # 1. 2-Hand Gestures (Mad vs Scared)
        if len(multi_hand_landmarks) == 2:
            h1 = multi_hand_landmarks[0].landmark
            h2 = multi_hand_landmarks[1].landmark
            
            # Average Height of Wrists
            avg_y = (h1[0].y + h2[0].y) / 2.0
            
            # Calculate Wrist Distance
            wrists_dist = np.sqrt((h1[0].x - h2[0].x)**2 + (h1[0].y - h2[0].y)**2)
            
            # Get Face Reference (Chin) if available
            face_bottom_y = 0.5 # Default fallback
            if face_landmarks:
                # Landmark 152 is usually the chin in MediaPipe Face Mesh
                face_bottom_y = face_landmarks.landmark[152].y

            # Logic: Scared (Home Alone) -> Hands generally open, near face height
            # Check if hands are roughly at or above the chin level
            # And check if fingers are extended (Open Palm looks more Scared)
            
            # Quick open palm check for both hands
            # We count extended fingers for each hand
            fingers_open_h1 = sum([self._is_finger_extended(h1, i, i-2) for i in [8, 12, 16, 20]])
            fingers_open_h2 = sum([self._is_finger_extended(h2, i, i-2) for i in [8, 12, 16, 20]])
            
            is_hands_open = (fingers_open_h1 >= 3) and (fingers_open_h2 >= 3)
            
            # Logic: Scared (Clasped Hands Low)
            # User request: "Clasp hands together below neck"
            # Condition: Wrists very close (touching/clasped) and below face
            if wrists_dist < 0.15 and avg_y > face_bottom_y:
                return "Scared"

            # Logic: Scared (Home Alone - High)
            # Existing logic: Hands near face height AND open palms
            if avg_y < (face_bottom_y + 0.1) and is_hands_open:
                return "Scared"
            
            # Logic: Mad (Folded Arms) -> LOW hands, Crossed, FISTS
            # Must be distinguished from Clasped Hands (Scared).
            # Folded arms usually have wrists separated (> 0.15).
            
            # Count open fingers for Mad check (should be low/fist)
            is_hands_fists = (fingers_open_h1 <= 1) and (fingers_open_h2 <= 1)
            
            if 0.15 <= wrists_dist < 0.5 and is_hands_fists:
                if avg_y > face_bottom_y:
                    return "Mad"

        # 2. Single Hand Gestures (Idea vs Think)
        for hand_landmarks in multi_hand_landmarks:
            lm = hand_landmarks.landmark
            
            # Index Extended?
            # Relaxed check for Think: Tip dist from MCP is large enough
            index_extended = self._is_finger_extended(lm, 8, 6)
            
            # Others Curled?
            middle_curled = not self._is_finger_extended(lm, 12, 10)
            ring_curled = not self._is_finger_extended(lm, 16, 14)
            pinky_curled = not self._is_finger_extended(lm, 20, 18)
            
            if index_extended and middle_curled and ring_curled and pinky_curled:
                # Potential Candidate for Idea or Think
                
                # --- FACE PROXIMITY CHECK (Prioritize this for Think) ---
                if face_landmarks:
                    # Mouth Center: Upper Lip (13) + Lower Lip (14) average
                    # Or just use Lower Lip (14)
                    mouth_x = face_landmarks.landmark[13].x
                    mouth_y = face_landmarks.landmark[13].y
                    
                    finger_tip_x = lm[8].x
                    finger_tip_y = lm[8].y
                    
                    dist_to_mouth = np.sqrt((finger_tip_x - mouth_x)**2 + (finger_tip_y - mouth_y)**2)
                    
                    # If very close to mouth, IT IS THINKING.
                    if dist_to_mouth < 0.15: # 15% of screen width/height? 
                        return "Think"
                        
                # --- FALLBACK / IDEA Logic ---
                
                # If not close to mouth, is it Idea?
                # Idea requires strict Verticality and Thumb Tuck.
                
                # Vertical check
                x_diff = abs(lm[8].x - lm[5].x)
                is_vertical = x_diff < 0.12 # Strict Vertical
                
                # Thumb check
                thumb_tip = lm[4]
                middle_mcp = lm[9]
                thumb_dist = np.sqrt((thumb_tip.x - middle_mcp.x)**2 + (thumb_tip.y - middle_mcp.y)**2)
                is_thumb_curled = thumb_dist < 0.2
                
                if is_vertical and is_thumb_curled:
                    return "Idea"
                    
                # If it wasn't vertical enough for Idea, and we didn't match face...
                # Maybe fallback to Think if high enough but no face detected?
                # Or just return None to avoid bugs.
                # User complaint: "Idea assumes I am doing think sometimes".
                # So we should be conservative with Think if no face is visible.
                
                if not face_landmarks:
                     # Fallback logic without face mesh
                     if lm[8].y < 0.35 and not is_vertical: # High but tilted?
                         return "Think"

        return None
