
# Add this to Echo's FastAPI service

@app.post("/api/vault/export-to-hashicorp")
async def export_vault_to_hashicorp():
    """Export Echo vault credentials to HashiCorp Vault"""
    try:
        # Import vault extension
        sys.path.append('/opt/tower-deepseek-api/echo-brain')
        from src.core.echo.echo_vault_extension import EchoVault
        
        # Initialize Echo vault
        echo_vault = EchoVault()
        credentials = echo_vault._load_credentials()
        
        # Initialize HashiCorp Vault
        import hvac
        vault_client = hvac.Client(url='http://127.0.0.1:8200')
        vault_client.token = open('/opt/vault/data/vault-token').read().strip()
        
        migrated_count = 0
        errors = []
        
        for service_name, cred_info in credentials.items():
            try:
                if isinstance(cred_info, dict) and 'data' in cred_info:
                    cred_type = cred_info.get('type', 'unknown')
                    cred_data = cred_info['data']
                    
                    # Determine storage location
                    if cred_type == 'oauth_provider' or 'oauth' in cred_type.lower():
                        # Store OAuth credentials
                        provider = service_name.replace('_oauth', '').replace('discovered_', '')
                        
                        vault_data = {
                            **cred_data,
                            'type': 'oauth_provider',
                            'status': 'migrated_from_echo',
                            'migrated_at': datetime.now().isoformat(),
                            'redirect_uri': cred_data.get('redirect_uri', f'https://localhost/api/auth/callback/{provider}')
                        }
                        
                        vault_client.secrets.kv.v2.create_or_update_secret(
                            path=provider,
                            secret=vault_data,
                            mount_point='oauth'
                        )
                        
                        migrated_count += 1
                        
                    else:
                        # Store API keys and other credentials
                        vault_data = {
                            **cred_data,
                            'type': cred_type,
                            'migrated_from_echo': True,
                            'migrated_at': datetime.now().isoformat()
                        }
                        
                        vault_client.secrets.kv.v2.create_or_update_secret(
                            path=f'echo/{service_name}',
                            secret=vault_data,
                            mount_point='services'
                        )
                        
                        migrated_count += 1
                        
            except Exception as e:
                errors.append(f"{service_name}: {str(e)}")
        
        return {
            "success": True,
            "migrated_count": migrated_count,
            "total_credentials": len(credentials),
            "errors": errors,
            "message": f"Successfully migrated {migrated_count} credentials to HashiCorp Vault"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to export vault credentials"
        }

@app.get("/api/vault/list-credentials")
async def list_vault_credentials():
    """List credentials available in Echo vault"""
    try:
        sys.path.append('/opt/tower-deepseek-api/echo-brain')
        from src.core.echo.echo_vault_extension import EchoVault
        
        echo_vault = EchoVault()
        credentials = echo_vault._load_credentials()
        
        cred_summary = []
        for service_name, cred_info in credentials.items():
            if isinstance(cred_info, dict):
                cred_summary.append({
                    'service': service_name,
                    'type': cred_info.get('type', 'unknown'),
                    'created_at': cred_info.get('created_at'),
                    'has_data': 'data' in cred_info and bool(cred_info['data'])
                })
        
        return {
            "success": True,
            "credentials": cred_summary,
            "total_count": len(cred_summary)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to list vault credentials"
        }
