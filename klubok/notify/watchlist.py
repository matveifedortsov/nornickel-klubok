"""Подписки на темы + лента уведомлений (§Y7 плана, требование ТЗ).

ТЗ («Дополнительные пожелания»): оповещение исследователя о появлении новых
публикаций/экспериментов по интересующей его теме. Реализация минимальная и
честная:

- подписка = (подписчик, тема-строка); хранение — SQLite рядом с аудитом
  (граф не место для пользовательского состояния);
- при ингесте документа новые имена сущностей + заголовок публикации матчатся
  против всех подписок; совпадения пишутся в ленту событий;
- логика матчинга (`match_topics`) — ЧИСТАЯ функция, тестируется без БД;
  нормализация терминов переиспользует отраслевой глоссарий RU/EN
  (klubok/extraction/resolver.canonical_id-стиль не нужен — берём сам GLOSSARY),
  чтобы «electrowinning» в новой статье срабатывал на подписку «электроэкстракция».
"""
from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from config import settings
from klubok.extraction.glossary_ru_en import GLOSSARY


def _canon_term(term: str) -> str:
    """Нормализовать термин для сравнения: lower/strip + отраслевой синоним."""
    t = " ".join(term.strip().lower().split())
    return GLOSSARY.get(t, t)


def match_topics(entity_names: list[str], topics: list[str]) -> list[tuple[str, str]]:
    """Чистая логика матчинга: какие темы задеты новыми сущностями.

    Возвращает список (topic, matched_entity). Совпадение — подстрока в любую
    сторону после нормализации (тема ⊆ сущность или сущность ⊆ тема), чтобы
    «католит» ловил «циркуляция католита», а «электроэкстракция никеля» —
    подписку «электроэкстракция». Синонимы RU/EN учитываются через глоссарий.
    """
    canon_entities = [(name, _canon_term(name)) for name in entity_names]
    hits: list[tuple[str, str]] = []
    for topic in topics:
        ct = _canon_term(topic)
        if not ct:
            continue
        for original, ce in canon_entities:
            if ct in ce or ce in ct:
                hits.append((topic, original))
                break
    return hits


@dataclass
class Notification:
    subscriber: str
    topic: str
    matched: str
    doc_id: str
    doc_title: str
    ts: str


class WatchStore:
    """SQLite-хранилище подписок и ленты уведомлений (потокобезопасное)."""

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path or settings.watchlist_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS subscriptions "
            "(subscriber TEXT, topic TEXT, created_at TEXT, "
            "PRIMARY KEY (subscriber, topic))")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS notifications "
            "(id INTEGER PRIMARY KEY AUTOINCREMENT, subscriber TEXT, topic TEXT, "
            "matched TEXT, doc_id TEXT, doc_title TEXT, ts TEXT, seen INT DEFAULT 0)")
        self._conn.commit()

    # --- подписки ---
    def subscribe(self, subscriber: str, topic: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO subscriptions VALUES (?, ?, ?)",
                (subscriber, topic, datetime.now(timezone.utc).isoformat()))
            self._conn.commit()

    def unsubscribe(self, subscriber: str, topic: str) -> None:
        with self._lock:
            self._conn.execute(
                "DELETE FROM subscriptions WHERE subscriber=? AND topic=?", (subscriber, topic))
            self._conn.commit()

    def topics(self) -> list[tuple[str, str]]:
        """Все (subscriber, topic) — для матчинга при ингесте."""
        cur = self._conn.execute("SELECT subscriber, topic FROM subscriptions")
        return [(r[0], r[1]) for r in cur.fetchall()]

    def subscriptions_of(self, subscriber: str) -> list[str]:
        cur = self._conn.execute(
            "SELECT topic FROM subscriptions WHERE subscriber=?", (subscriber,))
        return [r[0] for r in cur.fetchall()]

    # --- лента ---
    def _add_notification(self, n: Notification) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO notifications (subscriber, topic, matched, doc_id, doc_title, ts) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (n.subscriber, n.topic, n.matched, n.doc_id, n.doc_title, n.ts))
            self._conn.commit()

    def notify_new_document(self, doc_id: str, doc_title: str, entity_names: list[str]) -> int:
        """Сматчить новый документ против всех подписок, записать события в ленту.

        Возвращает число созданных уведомлений. Вызывается из
        pipeline.ingest_document после записи документа в граф.
        """
        subs = self.topics()
        if not subs:
            return 0
        by_subscriber: dict[str, list[str]] = {}
        for subscriber, topic in subs:
            by_subscriber.setdefault(subscriber, []).append(topic)

        count = 0
        ts = datetime.now(timezone.utc).isoformat()
        names = entity_names + ([doc_title] if doc_title else [])
        for subscriber, topics in by_subscriber.items():
            for topic, matched in match_topics(names, topics):
                self._add_notification(Notification(
                    subscriber=subscriber, topic=topic, matched=matched,
                    doc_id=doc_id, doc_title=doc_title, ts=ts))
                count += 1
        return count

    def feed(self, subscriber: str, limit: int = 50, unseen_only: bool = False) -> list[dict]:
        q = ("SELECT id, topic, matched, doc_id, doc_title, ts, seen FROM notifications "
             "WHERE subscriber=?" + (" AND seen=0" if unseen_only else "") +
             " ORDER BY id DESC LIMIT ?")
        cur = self._conn.execute(q, (subscriber, limit))
        cols = ["id", "topic", "matched", "doc_id", "doc_title", "ts", "seen"]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    def mark_seen(self, subscriber: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE notifications SET seen=1 WHERE subscriber=?", (subscriber,))
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()
