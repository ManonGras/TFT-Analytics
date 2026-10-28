from __future__ import annotations

from typing import Optional
from sqlalchemy import ForeignKey, create_engine, inspect, text
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    sessionmaker,
)

from config import DATABASE_PATH


class Base(DeclarativeBase):
    pass


class Match(Base):
    __tablename__ = "matches"

    match_id: Mapped[str] = mapped_column(primary_key=True)
    data_version: Mapped[Optional[str]]
    game_datetime: Mapped[Optional[int]]
    game_length: Mapped[Optional[float]]
    game_version: Mapped[Optional[str]]
    tft_set_number: Mapped[Optional[int]]
    tft_set_core_name: Mapped[Optional[str]]
    tft_game_type: Mapped[Optional[str]]
    queue_id: Mapped[Optional[int]]
    seed_tier: Mapped[Optional[str]]

    participants: Mapped[list["Participant"]] = relationship(
        back_populates="match", cascade="all, delete-orphan"
    )


class Participant(Base):
    __tablename__ = "participants"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    match_id: Mapped[str] = mapped_column(ForeignKey("matches.match_id"))
    puuid: Mapped[str]
    placement: Mapped[int]
    level: Mapped[Optional[int]]
    last_round: Mapped[Optional[int]]
    players_eliminated: Mapped[Optional[int]]
    time_eliminated: Mapped[Optional[float]]
    total_damage_to_players: Mapped[Optional[int]]
    gold_left: Mapped[Optional[int]]
    win: Mapped[Optional[bool]]

    match: Mapped["Match"] = relationship(back_populates="participants")
    units: Mapped[list["ParticipantUnit"]] = relationship(
        back_populates="participant", cascade="all, delete-orphan"
    )
    traits: Mapped[list["ParticipantTrait"]] = relationship(
        back_populates="participant", cascade="all, delete-orphan"
    )
    augments: Mapped[list["ParticipantAugment"]] = relationship(
        back_populates="participant", cascade="all, delete-orphan"
    )


class ParticipantUnit(Base):
    __tablename__ = "participant_units"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    participant_id: Mapped[int] = mapped_column(ForeignKey("participants.id"))
    character_id: Mapped[str]
    tier: Mapped[Optional[int]]
    rarity: Mapped[Optional[int]]

    participant: Mapped["Participant"] = relationship(back_populates="units")
    items: Mapped[list["ParticipantUnitItem"]] = relationship(
        back_populates="unit", cascade="all, delete-orphan"
    )


class ParticipantUnitItem(Base):
    __tablename__ = "participant_unit_items"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    unit_id: Mapped[int] = mapped_column(ForeignKey("participant_units.id"))
    item_name: Mapped[str]

    unit: Mapped["ParticipantUnit"] = relationship(back_populates="items")


class ParticipantTrait(Base):
    __tablename__ = "participant_traits"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    participant_id: Mapped[int] = mapped_column(ForeignKey("participants.id"))
    trait_name: Mapped[str]
    num_units: Mapped[Optional[int]]
    style: Mapped[Optional[int]]
    tier_current: Mapped[Optional[int]]
    tier_total: Mapped[Optional[int]]

    participant: Mapped["Participant"] = relationship(back_populates="traits")


class ParticipantAugment(Base):
    __tablename__ = "participant_augments"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    participant_id: Mapped[int] = mapped_column(ForeignKey("participants.id"))
    augment_name: Mapped[str]

    participant: Mapped["Participant"] = relationship(back_populates="augments")


def get_engine(path=DATABASE_PATH):
    return create_engine(f"sqlite:///{path}")


def init_db(path=DATABASE_PATH):
    engine = get_engine(path)
    Base.metadata.create_all(engine)

    inspector = inspect(engine)
    if "matches" in inspector.get_table_names():
        existing_columns = {col["name"] for col in inspector.get_columns("matches")}
        if "seed_tier" not in existing_columns:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE matches ADD COLUMN seed_tier VARCHAR"))

                conn.execute(text("UPDATE matches SET seed_tier = 'MASTER_PLUS' WHERE seed_tier IS NULL"))

    return engine


def get_session(path=DATABASE_PATH):
    engine = get_engine(path)
    Session = sessionmaker(bind=engine)
    return Session()


if __name__ == "__main__":
    engine = init_db()
    print(f"Database initialized: {DATABASE_PATH}")
    print("Tables created:", list(Base.metadata.tables.keys()))