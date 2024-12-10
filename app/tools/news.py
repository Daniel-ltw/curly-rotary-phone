from datetime import date
import json
from peewee import Model, TextField, IntegerField, DateField, SqliteDatabase, SQL, ForeignKeyField

db = SqliteDatabase("news-db.sqlite")

class BaseModel(Model):
    class Meta:
        database = db

class NewsGroup(BaseModel):
    id = IntegerField(primary_key=True)
    date = DateField(default=date.today())
    title = TextField()
    summary = TextField()
    keywords = TextField()

    class Meta:
        db_table = 'news_groups'

class News(BaseModel):
    id = IntegerField(primary_key=True)
    date = DateField(default=date.today())
    title = TextField()
    url = TextField()
    keywords = TextField()
    content = TextField()
    summary = TextField()
    source = TextField()
    group = ForeignKeyField(NewsGroup, backref='articles', null=True)

    class Meta:
        db_table = 'news'
        constraints = [
            # Define a composite unique constraint on date and title
            SQL('UNIQUE (date, title)')
        ]

    def llm_summary_format(self):
        return {self.title: self.summary.replace('ADVERTISEMENT', '').replace('\n\n', '\n')}

    def llm_content_format(self):
        return {self.title: self.content.replace('ADVERTISEMENT', '').replace('\n\n', '\n')}

    def llm_keywords_format(self):
        return {self.id: ", ".join(json.loads(self.keywords))}

class NewsSummary(BaseModel):
    id = IntegerField(primary_key=True)
    date = DateField(default=date.today())
    title = TextField()
    summary = TextField()
    keywords = TextField()

    class Meta:
        db_table = 'summaries'
        constraints = [
            SQL('UNIQUE (date, title)')
        ]
