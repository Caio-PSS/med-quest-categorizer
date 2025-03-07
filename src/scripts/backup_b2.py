# src/scripts/backup_b2.py
import logging
from datetime import datetime
from b2sdk.v2 import B2Api, InMemoryAccountInfo

def backup_to_b2():
    # Configuração
    B2_APPLICATION_KEY_ID = os.getenv('B2_KEY_ID')
    B2_APPLICATION_KEY = os.getenv('B2_APP_KEY')
    B2_BUCKET_NAME = 'med-quest-backup'
    
    try:
        # Inicializa cliente B2
        info = InMemoryAccountInfo()
        b2_api = B2Api(info)
        b2_api.authorize_account("production", B2_APPLICATION_KEY_ID, B2_APPLICATION_KEY)

        # Upload do arquivo
        bucket = b2_api.get_bucket_by_name(B2_BUCKET_NAME)
        file_name = f"backup-{datetime.now().strftime('%Y%m%d-%H%M')}.db"
        
        bucket.upload_local_file(
            local_file='/app/persistent_storage/questoes_medicas.db',
            file_name=file_name,
            progress_listener=None
        )
        
        logging.info(f"Backup {file_name} realizado com sucesso")
        
    except Exception as e:
        logging.error(f"Erro no backup B2: {str(e)}")
        raise

if __name__ == '__main__':
    backup_to_b2()