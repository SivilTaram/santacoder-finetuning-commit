from difflib import SequenceMatcher
from multiprocessing import Pool, Value
import os
import random
import re

import datasets
from huggingface_hub import hf_hub_download
import numpy as np
import json
from datasets import load_dataset, concatenate_datasets


GOOD_STARTS_EN = {'troubleshoot', 'indent', 'allow', 'access', 'load', 'mark', 'generalize', 'disable', 'merge',
                  'determine', 'rearrange', 'rectify', 'prepare', 'cut', 'edit', 'install', 'read', 'structure',
                  'recompile', 'debug', 'transform', 'orchestrate', 'develop', 'recomment', 'reset', 'validate',
                  'automate', 'indent', 'refresh', 'backup', 'replace', 'deal with', 'scrub', 'improve', 'terminate',
                  'monitor', 'revise', 'solve', 'split', 'designate', 'unstage', 'unwind', 'downgrade', 'handle',
                  'decommission', 'unify', 'add', 'reimplement', 'connect', 'archive', 'discard', 'compress', 'index',
                  'initialize', 'streamline', 'interpolate', 'format', 'append', 'delete', 'consolidate', 'settle',
                  'annotate', 'include', 'unblock', 'break', 'update', 'change', 'switch', 'reorganize', 'fix',
                  'reannotate', 'tackle', 'transpose', 'prepend', 'increase', 'integrate', 'order', 'reschedule',
                  'scale', 'maintain', 'truncate', 'drop', 'abort', 'remove', 'configure', 'unplug', 'save', 'create',
                  'reformat', 'rework', 'concatenate', 'decrypt', 'rewrite', 'check', 'divide', 'relocate', 'complete',
                  'dismantle', 'clarify', 'restructure', 'isolate', 'rollback', 'comment', 'send', 'standardize',
                  'clean', 'decompress', 'reword', 'provision', 'reorder', 'revoke', 'embed', 'redact', 'store',
                  'extend', 'unsync', 'return', 'optimize', 'align', 'test', 'reposition', 'package', 'simplify',
                  'tidy', 'establish', 'expire', 'deploy', 'plug ', 'reboot', 'enhance', 'attach', 'decrease',
                  'declare', 'rename', 'patch', 'print', 'rebuild', 'synchronize', 'trim', 'work', 'apply', 'copy',
                  'customize', 'expedite', 'call', 'purge', 'quit', 'unpublish', 'throw', 'clear', 'implement',
                  'define', 'make', 'watermark', 'raise', 'stop', 'substitute', 'normalize', 'rephrase', 'undo',
                  'paste', 'whitelist', 'mask', 'secure', 'rebase', 'set', 'tag', 'encrypt', 'reconnect', 'repackage',
                  'exit', 'arrange', 'build', 'migrate', 'swap', 'bring', 'bump', 'tweak', 'upgrade', 'write',
                  'resolve', 'put', 'exclude', 'insert', 'kill', 'subtract', 'repair', 'revert', 'redefine', 'enforce',
                  'convert', 'multiply', 'use', 'enable', 'support', 'document', 'correct', 'withdraw', 'move',
                  'modify', 'allot', 'introduce', 'address', 'increment', 'manage', 'verify', 'reconfigure', 'refactor'}
GOOD_STARTS_ZH = {'把', '替换', '降级', '保存', '修改', '解压缩', '撤销拉取', '修复', '对齐', '处理', '准备', '验证', '应用', '设置', '制作', '加速',
                  '校正', '补丁', '重新调度', '重新配置', '重新实现', '更改', '复制', '评论', '增强', '合并', '编排', '配置', '完成', '部署', '退出', '备份',
                  '回滚', '迁移', '添加', '减去', '重新排列', '重构', '重新定义', '拆分', '抛出', '串联', '简化流程', '终止', '取消暂存', '返回', '重新注释',
                  '标记', '重新排序', '插入', '插值', '修', '定制', '加密', '排除', '重写', '监控', '格式化', '撤销', '放弃', '掩码', '重新连接', '重新组织',
                  '清除', '追加', '停止', '建立索引', '解', '澄清', '微调', '重命名', '结束', '执行', '缩放', '取消发布', '乘以', '撤销暂存区的文件', '改善',
                  '丢弃', '归档', '重新编译', '解除同步', '注释', '解决', '拔掉', '包含', '简化', '清理', '变基', '删除', '同步', '介绍', '存档', '隔离',
                  '调试', '重新格式化', '重新定位', '中断', '转换', '结构化', '过期', '纠正', '刷新', '构建', '截断', '粘贴', '管理', '重新表述', '启用',
                  '整理', '改写', '支持', '文档化', '压缩', '检查', '白名单', '重新打包', '水印', '提高', '改进', '整合', '扩展', '升级', '重置', '移动',
                  '重建', '升级版本', '自动化', '测试', '修剪', '还原', '解除阻止', '剪切', '解决问题', '禁用', '修订', '维护', '解密', '标准化', '初始化',
                  '重新构建', '重启', '打包', '分割', '更新', '安全', '优化', '版本', '重新评论', '实现'}
GOOD_STARTS_FR = {'réinitialise', 'personnalise', 'masque', 'supporte', 'range', 'augmente', 'ajoute', 'liste blanche',
                  'structure', 'recompile', 'révise', 'désindexe', 'répare', 'retourne', 'soustrais', 'jette',
                  'complète', 'découpe', 'réimplémente', 'recommande', 'annote', 'débloque', 'indente', 'tague',
                  'réécris', 'sauvegarde', 'archive', 'reconnecte', 'supprime', 'vérifie', 'reconstruit', 'débranche',
                  'révoque', 'fabrique', 'refactorise', 'incrémente', 'désynchronise', 'prépare', 'change', 'dépanne',
                  'migre', 'implémente', 'introduit', 'édite', 'renomme', 'construis', 'ajoute un filigrane',
                  'convertit', 'travaille', 'configure', 'rectifie', 'clarifie', 'fournis', 'traite', 'transforme',
                  'aligne', 'sauve', 'réordonne', 'débogue', 'soulève', 'restructure', 'casse', 'reformule', 'modifie',
                  'rebases', 'prépend', 'réorganise', 'insère', 'rend anonyme', 'nettoie', 'chiffre', 'reconditionne',
                  'active', 'concatène', 'expire', 'définis', 'valide', 'formate', 'patche', 'compresse', 'commente',
                  'ordonne', 'fusionne', 'décompresse', 'normalise', 'emballe', 'repositionne', 'documente', 'applique',
                  'résous', 'met', 'maintient', 'résout', 'tronque', 'déplace', 'impose', 'désactive', 'réarrange',
                  'purge', 'isole', 'annule', 'corrige', 'arrête', 'adresse', 'multiplie', 'touche à', 'coupe',
                  'attaque', 'déploie', 'imprime', 'redéfinit', 'revient en arrière', 'échelle', 'régresse', 'unifie',
                  'ré-annote', 'exclue', 'améliore', 'rafraîchis', 'stocke', 'décrypte', 'colle', 'défait', 'remplace',
                  'automatise', 'simplifie', 'termine', 'reformate', 'branche', 'consolide', 'synchronise', 'redémarre',
                  'teste', 'tue', 'met à jour', 'divise', 'copie', 'défait le pull', 'gère', 'dépublie', 'intègre',
                  'étends', 'utilise', 'optimise', 'inclue', 'sécurise', 'retravaille', 'reviens en arrière',
                  'accélère', 'interpole', 'surveille', 'reconfigure', 'initialise', 'quitte', 'généralise',
                  'orchestre'}
GOOD_STARTS_ES = {'detiene', 'repara', 'revoca', 'reposiciona', 'termina', 'reconecta', 'depura', 'renombra',
                  'desbloquea', 'mejora', 'extiende', 'trabaja', 'aumenta', 'reorganiza', 'integra', 'reconstruye',
                  'eleva', 'documenta', 'transforma', 'devuelve', 'interpola', 'fusiona', 'implementa', 'rebasa',
                  'versiona', 'alinea', 'reimplementa', 'cifra', 'mantiene', 'prueba', 'refactoriza', 'toca',
                  'gestiona', 'sincroniza', 'rompe', 'migra', 'almacena', 'comprime', 'indenta', 'estructura',
                  'dessincroniza', 'incrementa', 'refresca', 'crea una rama', 'edita', 'inicializa', 'generaliza',
                  'lista blanca', 'cancela', 'soporta', 'modifica', 'concatena', 'oculta', 'soluciona', 'pega',
                  'desindexa', 'impone', 'personaliza', 'añade una marca de agua', 'recompila', 'deshace', 'provee',
                  'mueve', 'divide', 'copia', 'aisla', 'anota', 'unifica', 'recomienda', 'empaqueta', 'reordena',
                  'reanota', 'multiplica', 'elimina publicación', 'optimiza', 'guarda', 'valida', 'simplifica',
                  'acorta', 'asegura', 'aclara', 'normaliza', 'prepara', 'incluye', 'corta', 'retrabaja', 'reformatea',
                  'configura', 'reemplaza', 'descifra', 'reinicia', 'hace anónimo', 'reconfigura', 'revisa', 'acelera',
                  'completa', 'etiqueta', 'caduca', 'define', 'monitorea', 'actualiza', 'corrige', 'regresa',
                  'deshace el pull', 'descomprime', 'excluye', 'consolida', 'pone', 'verifica', 'reescribe',
                  'construye', 'imprime', 'desconecta', 'reformula', 'dirige', 'ataca', 'descarta', 'archiva',
                  'despliega', 'inserta', 'retrocede', 'mata', 'convierte', 'elimina', 'añade', 'ordena', 'sale',
                  'reestructura', 'vuelve atrás', 'comenta', 'recorta', 'cambia', 'activa', 'sustrae', 'desactiva',
                  'purga', 'procesa', 'redefine', 'limpia', 'escala', 'fabrica', 'formatea', 'automatiza', 'introduce',
                  'aplica', 'añade al principio', 'orquesta', 'resuelve', 'usa', 'parchea'}
GOOD_STARTS_PT = {'solucione', 'generalize', 'desconecte', 'combine', 'desative', 'reorganize', 'retifique', 'prepare',
                  'corte', 'edite', 'estruture', 'recompile', 'depure', 'transforme', 'recomente', 'orquestre', 'reset',
                  'valide', 'automatize', 'indente', 'atualize', 'faça backup', 'substitua', 'lidar com', 'limpe',
                  'melhore', 'termine', 'revisar', 'monitore', 'resolva', 'divida', 'remova do stage', 'faça downgrade',
                  'manuseie', 'unifique', 'adicione', 'reimplemente', 'arquive', 'inicialize', 'descarte', 'comprima',
                  'otimize', 'interpole', 'formate', 'anexe', 'delete', 'consolide', 'anote', 'inclua', 'desbloqueie',
                  'quebre', 'atualize', 'mude', 'reorganize', 'corrija', 'reanote', 'enfrente', 'prenda', 'dimensione',
                  'ordene', 'integre', 'remarque', 'mantenha', 'trunque', 'remova', 'aborte', 'teste', 'configure',
                  'salve', 'reformate', 'refaça', 'concatene', 'descriptografe', 'reescreva', 'verifique', 'divida',
                  'realoque', 'complete', 'clareie', 'reestruture', 'desfaça', 'isole', 'comente', 'padronize', 'limpe',
                  'descomprima', 'reformula', 'provisione', 'reordene', 'revogue', 'rediga', 'armazene',
                  'dessincronize', 'estenda', 'retorne', 'otimize', 'teste', 'alinhe', 'reposicione', 'arrume',
                  'simplifique', 'empacote', 'expire', 'conecte', 'implante', 'reinicie', 'melhore', 'renomeie',
                  'corrija', 'imprima', 'reconstrua', 'sincronize', 'aparar', 'trabalhe', 'aplique', 'copie',
                  'personalize', 'expedite', 'limpe', 'encerre', 'retire do ar', 'jogue', 'limpe', 'implante', 'faça',
                  'eleve', 'pare', 'reformule', 'normalize', 'desfaça', 'cole', 'liste', 'mascare', 'garanta', 'rebase',
                  'configure', 'marque', 'criptografe', 'reempacote', 'reconecte', 'saia', 'migre', 'construa',
                  'atualize', 'aumente', 'ajuste', 'resolva', 'coloque', 'exclua', 'mate', 'insira', 'subtraia',
                  'repare', 'reverta', 'redefina', 'imponha', 'converta', 'multiplique', 'use', 'ative', 'suporte',
                  'documente', 'corrija', 'mova', 'modifique', 'introduza', 'incremente'}
GOOD_STARTS_RU = {'упорядочить', 'добавить в конец', 'исправить', 'переформатировать', 'обрезать', 'удалить',
                  'проверить', 'упаковать', 'отладить', 'интегрировать', 'решить', 'протестировать',
                  'разделить на части', 'привести в порядок', 'объединить', 'сократить', 'добавить в начало',
                  'пересмотреть', 'уменьшить', 'прекратить', 'отозвать публикацию', 'синхронизировать', 'сжать',
                  'перепланировать', 'переименовать', 'настроить', 'истечь срок', 'отформатировать',
                  'перекомпилировать', 'структурировать', 'улучшить', 'обновлять', 'оркестрировать', 'отбросить',
                  'слияние', 'расшифровать', 'развернуть', 'вернуть', 'очистить', 'откатить', 'переставить',
                  'расширить', 'иметь дело с', 'перезагрузить', 'подключить', 'оптимизировать', 'прокомментировать',
                  'переработать', 'заменить', 'перестроить', 'отменить', 'заплатка', 'интерполировать',
                  'автоматизировать', 'добавить', 'стандартизировать', 'разделить', 'обновить', 'включить',
                  'переписать', 'отозвать', 'резервировать', 'склеить', 'сохранить', 'сбросить', 'предоставить',
                  'прервать', 'отслеживать', 'обработать', 'изменять', 'распаковать', 'переместить', 'понизить версию',
                  'архивировать', 'переорганизовать', 'исправить ошибку', 'отключить', 'устранить',
                  'выпустить новую версию', 'сделать что-то с', 'выровнять', 'изолировать', 'изменить порядок',
                  'отменить изменения', 'преобразовать', 'масштабировать', 'разблокировать', 'подготовить',
                  'инициализировать', 'переформулировать', 'завершить', 'поддерживать', 'перекомментировать',
                  'переосуществить'}
GOOD_STARTS_DE = {'verpacke', 'subtrahiere', 'formatiere', 'fabriziere', 'konfiguriere', 'benenne', 'implementiere',
                  'entsperre', 'verwalte', 'konvertiere', 'pflege', 'anonymisiere', 'widerrufe', 'skaliere', 'mache',
                  'entschlüssle', 'verringere', 'transformiere', 'archiviere', 'bereite vor', 'aktiviere', 'annotiere',
                  'verarbeite', 'definiere', 'aktualisiere', 'kopiere', 'schließe', 'normalisiere', 'unterstütze',
                  'arrangiere', 'kompiliere', 'rücke', 'inkrementiere', 'verschlüssele', 'speichere', 'beende', 'baue',
                  'stoppe', 'kehre zurück', 'mache', 'überarbeite', 'tagge', 'füge', 'setze zurück', 'starte neu',
                  'töte', 'bereinige', 'trenne', 'korrigiere', 'lösche', 'ordne', 'dokumentiere', 'hebe an',
                  'rekonstruiere', 'beschneide', 'multipliziere', 'empfehle', 'repariere', 'verzweige', 'maskiere',
                  'deaktiviere', 'vereinheitliche', 'kläre auf', 'schneide aus', 'zerbreche', 'personalisiere',
                  'werfe weg', 'berühre', 'patche', 'kommentiere', 'wende', 'führe ein', 'neu formuliere', 'liefere',
                  'behebe', 'vereinfache', 'positioniere', 'arbeite', 'ersetze', 'isoliere', 'refaktorisiere',
                  'fusioniere', 'erweitere', 'adressiere', 'validiere', 'depubliziere', 'gib', 'lass', 'verbessere',
                  'desynchronisiere', 'schreibe', 'ordne', 'vervollständige', 'setze', 'definiere', 'strukturiere',
                  'hebe', 'modifiziere', 'migriere', 'komprimiere', 'reimplementiere', 'ändere', 'setze', 'neubasiere',
                  'drucke', 'teste', 'ordne', 'organisiere', 'reformatiere', 'synchronisiere', 'deindexiere',
                  'verbinde', 'überprüfe', 'neu annotiere', 'teile', 'integriere', 'konsolidiere', 'dekomprimiere',
                  'strukturiere', 'klebe', 'richte', 'löse', 'verbinde', 'versioniere', 'bearbeite', 'erhöhe',
                  'erzwinge', 'automatisiere', 'bewege'}
GOOD_STARTS_KO = {'도입', '태그', '준비', '재수식하다', '확장하다', '재구현', '패치', '삭제', '권장', '비활성화하다', '재연결', '반환', '구현', '지원', '버리기',
                  '제공', '취소', '접근하다', '정규화', '재구성', '재작성', '인쇄', '변경', '생성', '재구조화', '포장', '화이트리스트', '주석', '문제를 해결하다',
                  '아카이브', '발생시키다', '재컴파일', '삽입', '개인화', '확인', '디버그', '검토', '정렬', '이름 바꾸기', '암호화', '증가', '취소하다', '연결하다',
                  '활성화', '인덱싱 해제', '곱하기', '완성', '명확하게', '연결 해제', '압축', '설치하다', '들여쓰기', '앞쪽에 추가하다', '작업', '재기반하다', '병합',
                  '범위', '잠금 해제', '리팩터링', '빼기', '제거', '워터마크 추가', '익명화', '동기화 해제', '적용', '분리', '구조', '만료', '압축 해제',
                  '되돌아가다', '편집', '저장', '유지 관리하다', '잘라내기', '추가', '재정의하다', '재배치', '중지', '해결', '보정', '빌드', '유효성 검사',
                  '다시 정렬하다', '사용하다', '재정렬', '깨다', '초기화', '적용하다', '최적화하다', '정의', '형식 지정', '이전', '설정', '변환', '문서화',
                  '수정하다', '가리개', '재조정', '이동하다', '수정', '정리', '배포', '수리', '주소', '처리', '버전 관리'}
GOOD_STARTS_JP = {'バックアップ', '移動', '設定', 'クリーンアップ', 'リフレッシュ', '保存', '再配置', 'バージョン', '再注釈', '再実装', '同期化', 'カスタマイズ', '対処',
                  '上げる', 'カット', '再語句化', '構造化', 'ステージの削除', '改善', '初期化', '維持', '再スケジュール', '戻る', '準備', '再編成', 'アンプラグ',
                  '改訂', '再フォーマット', '同期解除', '言い換え', 'チェック', 'トリミング', 'タグ', '置換', '暗号化', '再コンパイル', 'パッチ', 'モニター',
                  'トラブルシューティング', '自動化', '処理', '迅速化', '最適化', '再コメント', '単純化', 'インデックス化', '作成', '展開', '注釈', '再構築', '非公開化',
                  '補間', 'オーケストレーション', '強化', 'デバッグ', '切り捨て', '終了', 'リネーム', 'リベース', 'ブロック解除', 'ホワイトリスト', 'ロールバック',
                  'ダウングレード', '再起動', 'マージ', '提供', '拡張', 'クリア', '先頭に追加', 'アーカイブ', '並び替え', '修正', '整理', '明確化', '取り消し', '変更',
                  '期限切れ', '書き直し', '合理化', '破棄', '標準化', 'セキュア', '投げる', '復号', '適用', '完了', 'マスク', '貼り付け', '整列', 'パッケージ化',
                  'コピー', '統合', 'フォーマット', '結合', '解決', '無効化', '拡大', '実装', '削除', '検証', '圧縮', '中止', '再作業', '変換', '分離',
                  '透かし', 'テスト', '正規化', '含め', 'プラグイン', '解凍', '更新', '追加', 'コメント', '分割', '停止', 'リセット'}

# Add spaces to the end for languages that use spaces
GOOD_STARTS_EN = {word + " " for word in GOOD_STARTS_EN}
GOOD_STARTS_FR = {word + " " for word in GOOD_STARTS_FR}
GOOD_STARTS_ES = {word + " " for word in GOOD_STARTS_ES}
GOOD_STARTS_PT = {word + " " for word in GOOD_STARTS_PT}
GOOD_STARTS_RU = {word + " " for word in GOOD_STARTS_RU}
GOOD_STARTS_DE = {word + " " for word in GOOD_STARTS_DE}

# - In Japanese / Korean, the verb is usually at the end of the sentence
GOOD_ENDS_KO = {"세요"}
GOOD_ENDS_JP = {"て"}

GOOD_STARTS = GOOD_STARTS_EN | GOOD_STARTS_ZH | GOOD_STARTS_FR | GOOD_STARTS_ES | GOOD_STARTS_PT | GOOD_STARTS_RU | GOOD_STARTS_KO | GOOD_STARTS_JP
GOOD_ENDS = GOOD_ENDS_KO | GOOD_ENDS_JP

MODEL = "santacoder"
LANGUAGES = ["python", "java", "javascript"]

if MODEL == "bloomz":
    LANGUAGES += ["rust", "go", "c++"]
elif MODEL == "codegeex":
    # objective-c is the only one missing; Likely partly mixed in with C in the commits data
    LANGUAGES += [
        "rust", "go", "c++", "c", "html", "shell", "php", "html+php", "css", "typescript", "sql", "tex",
        "objective-c++", "scala", "kotlin", "pascal", "fortran", "r", "cuda", "c#"
    ]

# 1.0 mean keep all short commit messages
SHORT_SAMPLING = 1.0
LONG_SAMPLING = 1.0
LONG_SAMPLING_THRESHOLD = 500
# the ratio to control how many examples are fully shown in the model input, 0.2 means 20% examples will have
# the full code context such as the whole code file as the input
FULL_RANGE_FRAC = 0.5
# the minimum range and the maximum range represent the minimum context lines and the maximum context lines as the code context
MIN_RANGE = 0
MAX_RANGE = 32

NUM_PROC = 64

# take all the data
DATA_SAMPLING = 1.0
DATA_EXT = {"json", "yml", "xml", "html"}

BAD_SUB_MESSAGE = [
    "auto commit",
    "update contributing",
    "<?xml",
    "merge branch",
    "merge pull request",
]

BAD_MESSAGE = [
    "readme",
    "update",
    "dummy",
    "updated",
    # "debug",
    "test",
    "update readme",
    "update readme.md",
    "updated readme.md",
    "updated readme",
]
CACHE_DIR = "/dev/cache/liuqian/datasets/instruction-commits"
DATASET_NAME = "bigcode/instruction-commits"
PUSH_DATASET_NAME = "bigcode/commits-pjj-2048-0.5"
dataset_description = "This dataset is built with the following parameters: \n" \
                        f"The sampling parameters to balance the code modification range as:\n" \
                        f"  SHORT_SAMPLING: {SHORT_SAMPLING}\n" \
                        f"  LONG_SAMPLING: {LONG_SAMPLING}\n" \
                        f"  LONG_SAMPLING_THRESHOLD: {LONG_SAMPLING_THRESHOLD}\n" \
                        f"The sampling parameters to balance the programming language as:\n" \
                        f"  DATA_SAMPLING: {DATA_SAMPLING}\n" \
                        f"The sampling parameters to control the code context range as:\n" \
                        f"  FULL_RANGE_FRAC: {FULL_RANGE_FRAC}\n" \
                        f"  MIN_RANGE: {MIN_RANGE}\n" \
                        f"  MAX_RANGE: {MAX_RANGE}\n"

### SAMPLE ###
# BASE_DIR = "data"
# PATHS = [os.path.join(BASE_DIR, lang, f) for lang in LANGUAGES for f in os.listdir(BASE_DIR + "/" + lang)][:3]
# print(PATHS)

### FULL ###
counter = Value('i', 0)


def init(args):
    ''' store the counter for later use '''
    global counter
    counter = args


def prepare_download_files():
    downloaded_files = []
    for i in range(1, 459):
        downloaded_files.append("data/python/python-{:04d}.jsonl".format(i))

    for i in range(1, 517):
        downloaded_files.append("data/javascript/javascript-{:04d}.jsonl".format(i))

    for i in range(1, 250):
        downloaded_files.append("data/java/java-{:04d}.jsonl".format(i))
    return downloaded_files


data_files = prepare_download_files()


def download_file(file):
    global counter
    print("start")
    file = hf_hub_download(DATASET_NAME, file, repo_type="dataset",
                           cache_dir=CACHE_DIR)
    with counter.get_lock():
        counter.value += 1
    print(counter.value)
    return file


# download files using multi-thread
with Pool(16, initializer=init, initargs=(counter,)) as p:
    _ = p.map(download_file, data_files)

# obtain the file path
files = [hf_hub_download(DATASET_NAME, file, repo_type="dataset",
                         cache_dir=CACHE_DIR) for file in data_files]

counter = Value('i', 0)


def load_file(file):
    global counter
    print("start")
    file = load_dataset("/".join(file.split("/")[:-1]), data_files=file,
                        split="train", cache_dir=CACHE_DIR)
    with counter.get_lock():
        counter.value += 1
    print(counter.value)
    return file


with Pool(8, initializer=init, initargs=(counter,)) as p:
    ds_list = p.map(load_file, files)

ds = concatenate_datasets(ds_list)
print("The dataset size is: {}".format(len(ds)))


def get_line_diff_range(example):
    old_file_start = None
    old_file_end = 0

    new_file_start = None
    new_file_end = 0

    n_inserts = 0
    n_deletes = 0

    for group in SequenceMatcher(None, example["old_contents"].splitlines(),
                                 example["new_contents"].splitlines()).get_grouped_opcodes():
        group = [g for g in group if g[0] != "equal"]

        for element in group:
            if element[0] == "insert":
                n_inserts += element[4] - element[3]
            if element[0] == "delete":
                n_deletes += element[2] - element[1]
            if element[0] == "replace":
                n_deletes += element[2] - element[1]
                n_inserts += element[4] - element[3]

        first, last = group[0], group[-1]
        file1_range = (first[1], last[2])
        file2_range = (first[3], last[4])

        if old_file_start is None:
            old_file_start = file1_range[0]
        else:
            old_file_start = min(file1_range[0], old_file_start)
        old_file_end = max(file1_range[1], old_file_end)

        if new_file_start is None:
            new_file_start = file2_range[0]
        else:
            new_file_start = min(file2_range[0], new_file_start)
        new_file_end = max(file2_range[1], new_file_end)

    if old_file_start is None:
        old_file_start = 0
    if new_file_start is None:
        new_file_start = 0
    # -2 for compatibility with gh_diff
    example["old_change_start"] = old_file_start
    example["old_change_end"] = old_file_end
    example["old_change_range"] = old_file_end - old_file_start

    example["new_change_start"] = new_file_start
    example["new_change_end"] = new_file_end
    example["new_change_range"] = new_file_end - new_file_start

    example["n_inserts"] = n_inserts
    example["n_deletes"] = n_deletes
    example["n_changes"] = n_inserts + n_deletes

    return example


def clean_issues_and_refs(example):
    """
    Remove first word if 
        - [ ] in first word
        - : in first word

    Remove final word if
        - [ ] in final word
        - (# ) in final word

    E.g. 
    - [benchmark] Fix billing project (#9671) -> Fix billing project
    - demo/python/cmd.py: Fix struct.unpack format for Python 3 -> Fix struct.unpack format for Python 3
    """
    if len(example["subject"]) == 0:
        return example

    subject = example["subject"].split() + ["", "", ""]  # add empty strings to avoid index out of range

    if subject[0].startswith("[") and subject[0].endswith("]"):
        subject = subject[1:]

    if subject[0].endswith(":"):
        subject = subject[1:]

    if subject[-1].startswith("[") and subject[-1].endswith("]"):
        subject = subject[:-1]

    if "#" in subject[-1]:
        subject = subject[:-1]

    example["subject"] = " ".join(subject).strip()

    return example


# ds = ds.filter(lambda x: x["proba"] >= 0.9, num_proc=30)
# print("After proba filtering, the dataset size is: {}".format(len(ds)))

ds = ds.filter(lambda x: len(x["old_contents"]) < 100_000, num_proc=NUM_PROC)

print("After content length filtering, the dataset size is: {}".format(len(ds)))

ds = ds.filter(lambda x: len(x["new_contents"]) != 0, num_proc=NUM_PROC)

print("After empty new content filtering, the dataset size is: {}".format(len(ds)))


def filter_empty_messages(example):
    # Only filter out single alphabetic words (i.e. leave in e.g. Chinese)
    if len(example["subject"]) == 0 or (len(example["subject"].split()) == 1 and example["subject"].isalpha()):
        return False
    return True


ds = ds.filter(filter_empty_messages, num_proc=NUM_PROC)

print("After empty message filtering, the dataset size is: {}".format(len(ds)))

ds = ds.map(clean_issues_and_refs, num_proc=NUM_PROC)

ds = ds.filter(filter_empty_messages, num_proc=NUM_PROC)

print("After empty message filtering due to messages with []:, the dataset size is: {}".format(len(ds)))

ds = ds.map(get_line_diff_range, num_proc=NUM_PROC)


def filter_length(example):
    if example["old_change_range"] <= 2:
        if random.random() > SHORT_SAMPLING:
            return False

    if example["old_change_range"] >= LONG_SAMPLING_THRESHOLD:
        if random.random() > LONG_SAMPLING:
            return False

    return True


ds = ds.filter(filter_length, num_proc=NUM_PROC)


def filter_distribution(example):
    if example["old_file"].split(".")[-1] in DATA_EXT:
        if random.random() > DATA_SAMPLING:
            return False
    return True


ds = ds.filter(filter_distribution, num_proc=NUM_PROC)


def filter_messages(example):
    lower_subject = example["subject"].lower()

    # remove samples without desired start words or with low proba
    if not (lower_subject.startswith(tuple(GOOD_STARTS))) \
            and not (lower_subject.endswith(tuple(GOOD_ENDS))) \
            and (("proba" not in example) or (example["proba"] < 0.1)):
        return False

    # remove samples with bad messages
    if lower_subject in BAD_MESSAGE:
        return False

    # remove samples with bad subwords
    for bad_msg in BAD_SUB_MESSAGE:
        if bad_msg in lower_subject:
            return False

    # version updates (e.g. v1.1.0)
    if re.match(r"(?:v)?\d+\.\d+\.\d+(?=$|\S)", lower_subject):
        return False

    # commit message are hashes like 0239-2a41, but we do not want to remove english words like "debug"
    if re.match(r"^[a-f0-9]+(?:-[a-f0-9]+)*$", lower_subject):
        return False

    return True


ds = ds.filter(filter_messages, num_proc=NUM_PROC)

print("After message filtering, the dataset size is {}".format(len(ds)))

# Do not filter on HumanEval solutions as
# - Pre-training data is already filtered
# - Many of them are really basic code that can appear in different contexts
# - Commits data is from <2016 & HumanEval is human-created in 2021 ; OpenAI does also not decontaminate Codex data it seems
"""
from datasets import load_dataset
def load_dataset_column(dataset: str, column: str, split: str, name=None):
    ds = load_dataset(dataset, split=split, name=name)
    res = [sample[column].strip() for sample in ds]
    # Only return non-empty strings
    return [sample for sample in res if len(sample) > 0]

HUMAN_EVAL_STRINGS_OK = ['return x + y', 'return len(string)', 'return n**2', 'return ''.join(strings)']
human_eval_solutions = [s for s in load_dataset_column("openai_humaneval", "canonical_solution", "test") if s not in HUMAN_EVAL_STRINGS_OK]
"""


# Filter out HEXB docstrings https://github.com/bigcode-project/bigcode-dataset/blob/adb5fcd172b3015272d8ab976c0b53e95c804cd0/decontamination/find_substrings.py
def human_eval_docstrings():
    docstrings = []
    for lang in ["python", "java", "js", "go", "cpp", "rust"]:
        ds = datasets.load_dataset("bigcode/humaneval-x-bugs", lang)["test"]
        docstrings.extend([v['prompt'] for v in ds])
    return docstrings


human_eval_x_bugs_docstrings = human_eval_docstrings()
ds = ds.filter(lambda x: any([s in x["new_contents"] for s in human_eval_x_bugs_docstrings]) == False,
               num_proc=NUM_PROC)

print("After decontamination, the dataset size is {}".format(len(ds)))


def prepare_code(example):
    if np.random.random() < FULL_RANGE_FRAC:
        file_name = example["old_file"].split("/")[-1]
        code_before = example["old_contents"]
        code_after = example["new_contents"]
        example["content"] = f"<file_name>\n{file_name}\n<commit_before>\n{code_before}\n<commit_msg>\n{example['subject']}\n<commit_after>\n{code_after}"
        example["size"] = len(example["content"])
    else:
        start_offset = np.random.randint(MIN_RANGE, MAX_RANGE)
        end_offset = np.random.randint(MIN_RANGE, MAX_RANGE)

        old_lines = example["old_contents"].splitlines()
        new_lines = example["new_contents"].splitlines()

        old_start = max(0, example["old_change_start"] - start_offset)
        new_start = max(0, example["new_change_start"] - start_offset)

        old_end = min(len(old_lines), example["old_change_end"] + end_offset)
        new_end = min(len(new_lines), example["new_change_end"] + end_offset)

        code_before = "\n".join(old_lines[old_start:old_end])
        code_after = "\n".join(new_lines[new_start:new_end])

        example["old_contents"] = code_before
        example["new_contents"] = code_after
        file_name = example["old_file"].split("/")[-1]

        example["content"] = f"<file_name>\n{file_name}\n<commit_before>\n{code_before}\n<commit_msg>\n{example['subject']}\n<commit_after>\n{code_after}"
        example["size"] = len(example["content"])
    return example


def prepare_xp3(example):
    # input_template = "Instructions: {instruction}\nInput: {input} Output: "
    # example["inputs"] = f"Instructions: {example['subject']}\nInput: {example['old_contents']}"
    example["inputs"] = f"{example['old_contents']}\n\n{example['subject']}"
    example["targets"] = f"\n{example['new_contents']}"
    return example


if MODEL == "santacoder":
    ds = ds.map(prepare_code, num_proc=NUM_PROC)
elif MODEL == "bloomz":
    ds = ds.map(prepare_xp3, num_proc=NUM_PROC)

if MODEL == "santacoder":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bigcode/santacoder")
    # Filter for texts with with less than 2048 tokens
    ds = ds.filter(lambda x: len(tokenizer(x["content"])) <= 2048, num_proc=NUM_PROC)
elif MODEL == "bloomz":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-7b1")
    ds = ds.filter(
        lambda x: len(
            tokenizer(f"{x['old_contents']}\n\n{x['subject']}\n{x['new_contents']}")["input_ids"]) <= 2048,
        num_proc=NUM_PROC)
elif MODEL == "codegeex":
    from transformers.models.gpt2 import GPT2TokenizerFast

print("After length filtering, the dataset size is: {}".format(len(ds)))

if MODEL == "santacoder":
    cols_to_select = ["commit", "old_file", "new_file", "old_contents", "new_contents", "subject", "lang"] + [
        "proba"] if "proba" in ds.column_names else []
    cols_to_remove = [column_name for column_name in ds.column_names if column_name not in cols_to_select]
    print("Finally, the dataset size is {}".format(len(ds)))
    ds = ds.remove_columns(cols_to_remove)
    ds.push_to_hub(PUSH_DATASET_NAME, private=True)

    with open("dataset_description.json", "a+") as f:
        f.write(json.dumps({
            "dataset_size": len(ds),
            "dataset_name": PUSH_DATASET_NAME,
            "dataset_description": dataset_description
        }) + "\n")

elif MODEL == "bloomz":
    cols_to_select = ["inputs", "targets"]
    ds = ds.select_columns(cols_to_select)
    ds.to_json("commits.jsonl", orient="records", lines=True, force_ascii=False)
