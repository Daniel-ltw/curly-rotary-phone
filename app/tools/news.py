from datetime import date
from peewee import Model, TextField, IntegerField, DateField, SqliteDatabase, SQL

db = SqliteDatabase(f"db-{date.today()}.sqlite")

class BaseModel(Model):
    class Meta:
        database = db

class News(BaseModel):
    id = IntegerField(primary_key=True)
    date = DateField(default=date.today())
    title = TextField()
    url = TextField()
    keywords = TextField()
    content = TextField()
    summary = TextField()
    source = TextField()

    class Meta:
        db_table = 'news'
        constraints = [
            # Define a composite unique constraint on date and title
            SQL('UNIQUE (date, title)')
        ]

    def llm_summary_format(self):
        return {self.title: self.summary.replace('ADVERTISEMENT', '').replace('\n\n', '\n')} # :::{json.loads(self.keywords.decode())}

    def llm_content_format(self):
        return {self.title: self.content.replace('ADVERTISEMENT', '').replace('\n\n', '\n')} # :::{json.loads(self.keywords.decode())}

class NewsSummary(BaseModel):
    id = IntegerField(primary_key=True)
    date = DateField(default=date.today())
    title = TextField()
    summary = TextField()

    class Meta:
        db_table = 'summaries'
        constraints = [
            SQL('UNIQUE (date, title)')
        ]
