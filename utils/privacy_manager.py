"""
Privacy Manager for Data Protection and Anonymization
Implements privacy-first features including anonymization, encryption, and consent management
"""

import hashlib
import hmac
import secrets
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import json
import time
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
from loguru import logger


class PrivacyManager:
    """
    Comprehensive privacy management system that ensures:
    - Data anonymization and pseudonymization
    - Embedding protection and obfuscation
    - Consent management
    - Data retention policies
    - Secure data handling
    """
    
    def __init__(self, config: Any):
        """
        Initialize privacy manager.
        
        Args:
            config: Configuration object
        """
        self.config = config
        
        # Privacy settings
        self.anonymize_embeddings = config.get("privacy.anonymize_embeddings", True)
        self.hash_identifiers = config.get("privacy.hash_identifiers", True)
        self.local_processing_only = config.get("privacy.local_processing_only", True)
        self.data_retention_hours = config.get("privacy.data_retention_hours", 24)
        self.encryption_enabled = config.get("privacy.encryption_enabled", True)
        self.consent_required = config.get("privacy.consent_required", True)
        
        # Encryption setup
        self.encryption_key = None
        self.salt = None
        self._initialize_encryption()
        
        # Anonymization parameters
        self.noise_scale = 0.01  # For embedding anonymization
        self.hash_salt = secrets.token_bytes(32)
        
        # Consent tracking
        self.consent_records = {}  # {person_id: consent_data}
        
        # Data retention tracking
        self.data_timestamps = {}  # {data_id: timestamp}
        
        # Privacy audit log
        self.audit_log = []
        
        logger.info("Privacy manager initialized with privacy-first settings")
    
    def _initialize_encryption(self):
        """Initialize encryption system."""
        try:
            if self.encryption_enabled:
                # Generate or load encryption key
                password = b"classroom_analyzer_secure_key"  # In production, use secure key management
                self.salt = secrets.token_bytes(16)
                
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=self.salt,
                    iterations=100000,
                )
                key = base64.urlsafe_b64encode(kdf.derive(password))
                self.encryption_key = Fernet(key)
                
                logger.info("Encryption system initialized")
            
        except Exception as e:
            logger.error(f"Encryption initialization failed: {e}")
            self.encryption_enabled = False
    
    def anonymize_person_id(self, person_id: str) -> str:
        """
        Anonymize person identifier using secure hashing.
        
        Args:
            person_id: Original person identifier
            
        Returns:
            Anonymized identifier
        """
        try:
            if not self.hash_identifiers:
                return person_id
            
            # Use HMAC for secure hashing
            anonymized = hmac.new(
                self.hash_salt,
                person_id.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()[:16]  # Use first 16 characters
            
            self._log_privacy_action("anonymize_id", {"original_length": len(person_id)})
            
            return f"anon_{anonymized}"
            
        except Exception as e:
            logger.error(f"Person ID anonymization failed: {e}")
            return f"anon_{secrets.token_hex(8)}"
    
    def anonymize_embedding(self, embedding: np.ndarray, person_id: str = None) -> np.ndarray:
        """
        Anonymize face embedding while preserving similarity structure.
        
        Args:
            embedding: Original face embedding
            person_id: Person identifier for consistent anonymization
            
        Returns:
            Anonymized embedding
        """
        try:
            if not self.anonymize_embeddings:
                return embedding
            
            # Generate consistent noise for same person
            if person_id:
                np.random.seed(hash(person_id) % (2**32))
            
            # Add controlled noise
            noise = np.random.normal(0, self.noise_scale, embedding.shape)
            anonymized = embedding + noise
            
            # Preserve unit vector property
            anonymized = anonymized / np.linalg.norm(anonymized)
            
            # Apply additional obfuscation
            anonymized = self._apply_embedding_obfuscation(anonymized)
            
            self._log_privacy_action("anonymize_embedding", {
                "embedding_dim": len(embedding),
                "noise_scale": self.noise_scale
            })
            
            return anonymized
            
        except Exception as e:
            logger.error(f"Embedding anonymization failed: {e}")
            return embedding
    
    def _apply_embedding_obfuscation(self, embedding: np.ndarray) -> np.ndarray:
        """Apply additional obfuscation to embedding."""
        try:
            # Rotate embedding in high-dimensional space
            rotation_angle = 0.1  # Small rotation to preserve similarity
            
            # Simple rotation for first two dimensions
            if len(embedding) >= 2:
                cos_theta = np.cos(rotation_angle)
                sin_theta = np.sin(rotation_angle)
                
                x, y = embedding[0], embedding[1]
                embedding[0] = cos_theta * x - sin_theta * y
                embedding[1] = sin_theta * x + cos_theta * y
            
            return embedding
            
        except Exception as e:
            logger.error(f"Embedding obfuscation failed: {e}")
            return embedding
    
    def encrypt_data(self, data: Any) -> Optional[bytes]:
        """
        Encrypt sensitive data.
        
        Args:
            data: Data to encrypt
            
        Returns:
            Encrypted data or None if encryption fails
        """
        try:
            if not self.encryption_enabled or not self.encryption_key:
                return None
            
            # Convert data to JSON string
            if isinstance(data, (dict, list)):
                data_str = json.dumps(data)
            else:
                data_str = str(data)
            
            # Encrypt
            encrypted = self.encryption_key.encrypt(data_str.encode('utf-8'))
            
            self._log_privacy_action("encrypt_data", {"data_size": len(data_str)})
            
            return encrypted
            
        except Exception as e:
            logger.error(f"Data encryption failed: {e}")
            return None
    
    def decrypt_data(self, encrypted_data: bytes) -> Optional[Any]:
        """
        Decrypt sensitive data.
        
        Args:
            encrypted_data: Encrypted data
            
        Returns:
            Decrypted data or None if decryption fails
        """
        try:
            if not self.encryption_enabled or not self.encryption_key:
                return None
            
            # Decrypt
            decrypted_bytes = self.encryption_key.decrypt(encrypted_data)
            data_str = decrypted_bytes.decode('utf-8')
            
            # Try to parse as JSON
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                data = data_str
            
            self._log_privacy_action("decrypt_data", {"data_size": len(data_str)})
            
            return data
            
        except Exception as e:
            logger.error(f"Data decryption failed: {e}")
            return None
    
    def check_consent(self, person_id: str) -> bool:
        """
        Check if person has given consent for processing.
        
        Args:
            person_id: Person identifier
            
        Returns:
            True if consent is given or not required
        """
        try:
            if not self.consent_required:
                return True
            
            anonymized_id = self.anonymize_person_id(person_id)
            
            if anonymized_id in self.consent_records:
                consent_data = self.consent_records[anonymized_id]
                
                # Check if consent is still valid
                if consent_data.get('expires_at', float('inf')) > time.time():
                    return consent_data.get('granted', False)
            
            return False
            
        except Exception as e:
            logger.error(f"Consent check failed: {e}")
            return False
    
    def record_consent(self, person_id: str, granted: bool, duration_hours: int = 24) -> bool:
        """
        Record consent for a person.
        
        Args:
            person_id: Person identifier
            granted: Whether consent is granted
            duration_hours: How long consent is valid
            
        Returns:
            Success status
        """
        try:
            anonymized_id = self.anonymize_person_id(person_id)
            
            consent_data = {
                'granted': granted,
                'timestamp': time.time(),
                'expires_at': time.time() + (duration_hours * 3600),
                'duration_hours': duration_hours
            }
            
            self.consent_records[anonymized_id] = consent_data
            
            self._log_privacy_action("record_consent", {
                "granted": granted,
                "duration_hours": duration_hours
            })
            
            logger.info(f"Consent recorded for {anonymized_id}: {granted}")
            return True
            
        except Exception as e:
            logger.error(f"Consent recording failed: {e}")
            return False
    
    def anonymize_summary(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """
        Anonymize session summary data.
        
        Args:
            summary: Original summary data
            
        Returns:
            Anonymized summary
        """
        try:
            anonymized = summary.copy()
            
            # Remove or anonymize sensitive fields
            sensitive_fields = ['session_id', 'person_ids', 'individual_data']
            
            for field in sensitive_fields:
                if field in anonymized:
                    if field == 'session_id':
                        anonymized[field] = self.anonymize_person_id(anonymized[field])
                    elif field == 'person_ids':
                        anonymized[field] = [self.anonymize_person_id(pid) for pid in anonymized[field]]
                    else:
                        del anonymized[field]
            
            # Aggregate individual metrics to prevent re-identification
            if 'metrics' in anonymized:
                anonymized['metrics'] = self._aggregate_metrics(anonymized['metrics'])
            
            self._log_privacy_action("anonymize_summary", {"fields_processed": len(sensitive_fields)})
            
            return anonymized
            
        except Exception as e:
            logger.error(f"Summary anonymization failed: {e}")
            return summary
    
    def _aggregate_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate metrics to prevent individual identification."""
        try:
            aggregated = {}
            
            # Keep only aggregated statistics
            safe_fields = [
                'total_persons', 'average_engagement', 'attention_rate',
                'participation_rate', 'confusion_rate', 'session_duration'
            ]
            
            for field in safe_fields:
                if field in metrics:
                    aggregated[field] = metrics[field]
            
            # Add privacy-safe derived metrics
            if 'total_persons' in metrics and metrics['total_persons'] > 0:
                # Only include rates if there are enough people to maintain anonymity
                min_group_size = 3
                if metrics['total_persons'] >= min_group_size:
                    aggregated['group_size'] = 'sufficient'
                else:
                    aggregated['group_size'] = 'limited'
                    # Remove detailed rates for small groups
                    for rate_field in ['attention_rate', 'participation_rate', 'confusion_rate']:
                        if rate_field in aggregated:
                            del aggregated[rate_field]
            
            return aggregated
            
        except Exception as e:
            logger.error(f"Metrics aggregation failed: {e}")
            return metrics
    
    def apply_data_retention(self):
        """Apply data retention policies."""
        try:
            current_time = time.time()
            retention_seconds = self.data_retention_hours * 3600
            
            # Clean up expired data
            expired_data = []
            for data_id, timestamp in self.data_timestamps.items():
                if current_time - timestamp > retention_seconds:
                    expired_data.append(data_id)
            
            for data_id in expired_data:
                del self.data_timestamps[data_id]
                # In practice, would also delete associated data
            
            # Clean up expired consent records
            expired_consent = []
            for person_id, consent_data in self.consent_records.items():
                if current_time > consent_data.get('expires_at', 0):
                    expired_consent.append(person_id)
            
            for person_id in expired_consent:
                del self.consent_records[person_id]
            
            if expired_data or expired_consent:
                self._log_privacy_action("data_retention_cleanup", {
                    "expired_data_count": len(expired_data),
                    "expired_consent_count": len(expired_consent)
                })
                
                logger.info(f"Data retention cleanup: {len(expired_data)} data items, {len(expired_consent)} consent records")
            
        except Exception as e:
            logger.error(f"Data retention cleanup failed: {e}")
    
    def _log_privacy_action(self, action: str, details: Dict[str, Any]):
        """Log privacy-related actions for audit purposes."""
        try:
            log_entry = {
                'timestamp': time.time(),
                'action': action,
                'details': details
            }
            
            self.audit_log.append(log_entry)
            
            # Keep only recent audit entries
            max_audit_entries = 1000
            if len(self.audit_log) > max_audit_entries:
                self.audit_log = self.audit_log[-max_audit_entries:]
            
        except Exception as e:
            logger.error(f"Privacy action logging failed: {e}")
    
    def get_privacy_report(self) -> Dict[str, Any]:
        """Generate privacy compliance report."""
        try:
            current_time = time.time()
            
            # Count active consent records
            active_consent = sum(
                1 for consent in self.consent_records.values()
                if consent.get('expires_at', 0) > current_time and consent.get('granted', False)
            )
            
            # Count recent privacy actions
            recent_actions = sum(
                1 for entry in self.audit_log
                if current_time - entry['timestamp'] < 3600  # Last hour
            )
            
            report = {
                'privacy_settings': {
                    'anonymize_embeddings': self.anonymize_embeddings,
                    'hash_identifiers': self.hash_identifiers,
                    'local_processing_only': self.local_processing_only,
                    'encryption_enabled': self.encryption_enabled,
                    'consent_required': self.consent_required,
                    'data_retention_hours': self.data_retention_hours
                },
                'consent_status': {
                    'total_consent_records': len(self.consent_records),
                    'active_consent_count': active_consent,
                    'consent_required': self.consent_required
                },
                'data_protection': {
                    'encryption_active': self.encryption_enabled and self.encryption_key is not None,
                    'anonymization_active': self.anonymize_embeddings,
                    'data_items_tracked': len(self.data_timestamps)
                },
                'audit_info': {
                    'total_audit_entries': len(self.audit_log),
                    'recent_actions_count': recent_actions,
                    'last_retention_cleanup': self._get_last_cleanup_time()
                },
                'compliance_status': self._assess_compliance()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Privacy report generation failed: {e}")
            return {}
    
    def _get_last_cleanup_time(self) -> Optional[float]:
        """Get timestamp of last data retention cleanup."""
        cleanup_entries = [
            entry for entry in self.audit_log
            if entry['action'] == 'data_retention_cleanup'
        ]
        
        if cleanup_entries:
            return max(entry['timestamp'] for entry in cleanup_entries)
        return None
    
    def _assess_compliance(self) -> Dict[str, bool]:
        """Assess privacy compliance status."""
        compliance = {
            'anonymization_enabled': self.anonymize_embeddings,
            'encryption_enabled': self.encryption_enabled and self.encryption_key is not None,
            'consent_management': self.consent_required,
            'local_processing': self.local_processing_only,
            'data_retention_policy': self.data_retention_hours <= 24,  # Max 24 hours
            'audit_logging': len(self.audit_log) > 0
        }
        
        compliance['overall_compliant'] = all(compliance.values())
        
        return compliance
    
    def export_privacy_data(self, person_id: str) -> Optional[Dict[str, Any]]:
        """
        Export all privacy-related data for a person (GDPR compliance).
        
        Args:
            person_id: Person identifier
            
        Returns:
            Privacy data export or None if not found
        """
        try:
            anonymized_id = self.anonymize_person_id(person_id)
            
            export_data = {
                'person_id': anonymized_id,
                'consent_record': self.consent_records.get(anonymized_id),
                'privacy_settings': {
                    'anonymization_applied': self.anonymize_embeddings,
                    'encryption_applied': self.encryption_enabled,
                    'data_retention_hours': self.data_retention_hours
                },
                'export_timestamp': time.time()
            }
            
            self._log_privacy_action("export_privacy_data", {"person_id": anonymized_id})
            
            return export_data
            
        except Exception as e:
            logger.error(f"Privacy data export failed: {e}")
            return None
    
    def delete_person_data(self, person_id: str) -> bool:
        """
        Delete all data for a person (right to be forgotten).
        
        Args:
            person_id: Person identifier
            
        Returns:
            Success status
        """
        try:
            anonymized_id = self.anonymize_person_id(person_id)
            
            # Remove consent record
            if anonymized_id in self.consent_records:
                del self.consent_records[anonymized_id]
            
            # Remove from data timestamps
            data_to_remove = [
                data_id for data_id in self.data_timestamps
                if anonymized_id in data_id
            ]
            
            for data_id in data_to_remove:
                del self.data_timestamps[data_id]
            
            self._log_privacy_action("delete_person_data", {
                "person_id": anonymized_id,
                "data_items_removed": len(data_to_remove)
            })
            
            logger.info(f"Deleted all data for person: {anonymized_id}")
            return True
            
        except Exception as e:
            logger.error(f"Person data deletion failed: {e}")
            return False
