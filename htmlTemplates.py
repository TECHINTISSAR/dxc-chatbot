
css = '''

<style>
[data-testid="stAppViewContainer"]{
    background-color:#d8bdd8
}
    background-color: #000000;

.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #e0d7e0
}
.chat-message.bot {
    background-color: #C285C2
}
.chat-message .avatar {
  width: 20%;
      margin-right: 1rem;  /* Adds space between the avatar and message */

}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
.chat-history-frame {
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 15px;
    margin-bottom: 15px;
    background-color: #fff;
}


'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
<a href="https://imgbb.com/"><img src="https://i.ibb.co/R3zQKpF/Capture-d-cran-2024-05-23-150205.png" alt="Capture-d-cran-2024-05-23-150205" border="0"></a>    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
<a href="https://imgbb.com/"><img src="https://i.ibb.co/VVHVcqz/de.png" alt="de" border="0"></a>    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''
